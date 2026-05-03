"""
Парсер вакансий с SuperJob.ru (HTML-скрапинг, без API-ключа).

Примечание: резюме на SuperJob доступны только авторизованным работодателям
и рендерятся на клиенте — парсинг резюме без авторизации невозможен.
"""
import requests
import time
import re
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_MSK = ZoneInfo('Europe/Moscow')


def _now_msk() -> str:
    return datetime.now(_MSK).strftime('%Y-%m-%d %H:%M:%S')


class SJVacancyParser:
    """Парсер вакансий с SuperJob.ru."""

    SJ_AREAS = {
        'Москва': 4,
        'Санкт-Петербург': 14,
        'Екатеринбург': 9,
        'Новосибирск': 8,
        'Нижний Новгород': 19,
        'Казань': 24,
        'Самара': 20,
        'Омск': 11,
        'Челябинск': 23,
        'Ростов-на-Дону': 16,
        'Уфа': 6,
        'Красноярск': 7,
        'Воронеж': 21,
        'Пермь': 13,
        'Волгоград': 22,
        'Краснодар': 93,
        'Саратов': 17,
        'Тюмень': 10,
        'Тольятти': 27,
        'Ижевск': 26,
        'Барнаул': 18,
        'Владивосток': 57,
        'Иркутск': 33,
        'Хабаровск': 55,
        'Ярославль': 25,
        'Томск': 35,
        'Оренбург': 56,
        'Кемерово': 39,
        'Рязань': 41,
        'Астрахань': 43,
        'Пенза': 45,
        'Липецк': 48,
        'Киров': 47,
        'Тула': 50,
        'Ставрополь': 53,
        'Белгород': 64,
        'Владимир': 68,
        'Смоленск': 71,
        'Калуга': 73,
        'Вологда': 78,
        'Мурманск': 79,
        'Россия': 0,
    }

    def __init__(self, timeout=30, max_retries=3, max_404_errors=5):
        self.base_url = 'https://www.superjob.ru'
        self.search_url = 'https://www.superjob.ru/vakansii/'
        self.timeout = timeout
        self.consecutive_404_errors = 0
        self.max_404_errors = max_404_errors

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET'],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session = requests.Session()
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.superjob.ru/',
        })

    # ── 404 / error handling ──────────────────────────────────────────

    def _check_404_limit(self):
        return self.consecutive_404_errors >= self.max_404_errors

    def _handle_404_error(self):
        self.consecutive_404_errors += 1
        if self._check_404_limit():
            raise StopIteration('Превышен лимит ошибок 404')

    def _handle_successful_request(self):
        if self.consecutive_404_errors > 0:
            self.consecutive_404_errors = 0

    # ── Search ───────────────────────────────────────────────────────

    def search_vacancies(self, search_text: str, area: int = 4, page: int = 1) -> str | None:
        """Возвращает HTML страницы поиска вакансий или None."""
        if self._check_404_limit():
            return None

        params = {
            'keywords': search_text,
            'town': area,
        }
        if page > 1:
            params['page'] = page

        try:
            response = self.session.get(
                self.search_url, params=params, timeout=self.timeout, allow_redirects=True,
            )
            if response.status_code == 200:
                self._handle_successful_request()
                return response.text
            elif response.status_code == 404:
                self._handle_404_error()
                return None
            else:
                self._handle_successful_request()
                print(f'[{response.status_code}] {response.text[:200]}')
                return None
        except StopIteration:
            raise
        except Exception:
            self._handle_successful_request()
            return None

    # ── Parse search results ─────────────────────────────────────────

    def parse_vacancies_page(self, html: str) -> list[dict]:
        """Извлекает список вакансий из HTML-страницы поиска."""
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        # Ссылки на конкретные вакансии — slug + числовой ID в URL
        job_links = soup.find_all('a', href=re.compile(r'/vakansii/.+-\d{6,}\.html'))

        seen_urls, vacancies = set(), []
        for link in job_links:
            href = link.get('href', '')
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)

            id_match = re.search(r'-(\d+)\.html$', href)
            if not id_match:
                continue

            url = urljoin(self.base_url, href)
            details = self.get_vacancy_details(id_match.group(1), url)
            if details:
                vacancies.append(details)
            time.sleep(1)

        return vacancies

    # ── Detail page ───────────────────────────────────────────────────

    def get_vacancy_details(self, vacancy_id: str, url: str, max_retries: int = 2) -> dict | None:
        """Разбирает детальную страницу вакансии SuperJob."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return self._extract_detail(vacancy_id, url, response.text)
                elif response.status_code == 404:
                    return None
                elif response.status_code in (403, 429):
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                    continue
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _extract_detail(self, vacancy_id: str, url: str, html: str) -> dict:
        soup = BeautifulSoup(html, 'html.parser')
        page_text = soup.get_text(separator='\n', strip=True)

        # Название вакансии
        h1 = soup.find('h1')
        name = h1.get_text(strip=True) if h1 else ''

        # Работодатель — первый крупный текстовый блок после h1 без ₽ и цифр
        employer = self._extract_employer(soup, page_text)

        # Местоположение
        area = self._extract_area(page_text)

        # Зарплата
        salary = self._extract_salary(soup)

        # Поля Label:Value из текста страницы
        employment = self._extract_field(page_text, r'Тип\s+занятости[:\s]+(.+?)(?:\n|$)')
        schedule_raw = self._extract_field(page_text, r'График\s+работы[:\s]+(.+?)(?:\n|$)')
        schedule = self._map_schedule(schedule_raw)
        experience = self._extract_field(page_text, r'Опыт\s+работы[:\s]*(.+?)(?:\n|$)')

        # Описание — ищем блоки без вложенных div/section (листья DOM)
        description = self._extract_description(soup)

        # Требования и обязанности — из описания по маркерам
        requirement = self._extract_section(description, ['Требования', 'Обязательно', 'Ключевые требования'])
        responsibility = self._extract_section(description, ['Обязанности', 'Ключевые обязанности', 'Чем предстоит заниматься'])

        # Дата публикации
        published_at = self._extract_date(page_text)

        return {
            'id':              vacancy_id,
            'name':            name,
            'area':            area,
            'url':             url,
            'alternate_url':   url,
            'requirement':     requirement,
            'responsibility':  responsibility,
            'description':     description,
            'employer':        employer,
            'experience':      experience,
            'employment':      employment,
            'schedule':        schedule,
            'published_at':    published_at,
            'created_at':      published_at,
            'salary_from':     salary['from'],
            'salary_to':       salary['to'],
            'salary_currency': salary['currency'],
            'salary_gross':    salary['gross'],
            'source':          'superjob.ru',
        }

    # ── Field extractors ──────────────────────────────────────────────

    def _extract_field(self, page_text: str, pattern: str) -> str:
        m = re.search(pattern, page_text, re.I)
        return m.group(1).strip() if m else ''

    def _extract_employer(self, soup: BeautifulSoup, page_text: str) -> str:
        # Из тега <title>: "...работа в компании НАЗВАНИЕ (вакансия №..."
        # Останавливаемся на первой "(", ",", " — ", " | " или "на SuperJob"
        title = soup.find('title')
        if title:
            m = re.search(
                r'работа в компании\s+([^(,\n]+?)(?:\s*[,(—|]|\s+на\s+SuperJob|$)',
                title.get_text(), re.I,
            )
            if m:
                return m.group(1).strip().rstrip(' ,.-—')
        return ''

    def _extract_area(self, page_text: str) -> str:
        cities = '|'.join(list(self.SJ_AREAS.keys())[:-1])  # без "Россия"
        m = re.search(rf'({cities})(?:,\s*[^\n]{{0,60}})?', page_text)
        return m.group(1).strip() if m else ''

    def _extract_salary(self, soup: BeautifulSoup) -> dict:
        result = {'from': None, 'to': None, 'currency': None, 'gross': None}
        # Ищем span с ₽ вблизи h1 (первое вхождение на странице)
        for elem in soup.find_all(string=re.compile(r'₽|руб')):
            text = elem.strip()
            if 2 < len(text) < 60:
                parsed = self._parse_salary(text)
                if parsed['currency']:
                    return parsed
        return result

    def _parse_salary(self, salary_text: str) -> dict:
        result = {'from': None, 'to': None, 'currency': None, 'gross': None}
        if not salary_text:
            return result

        if '₽' in salary_text or 'руб' in salary_text.lower():
            result['currency'] = 'RUR'
        elif '$' in salary_text or 'USD' in salary_text:
            result['currency'] = 'USD'
        elif '€' in salary_text or 'EUR' in salary_text:
            result['currency'] = 'EUR'

        # Убираем пробелы внутри чисел и собираем все числа
        clean = re.sub(r'(\d)\s+(\d)', r'\1\2', salary_text)
        numbers = [int(n) for n in re.findall(r'\d{4,}', clean)]

        lower = salary_text.lower()
        if 'до' in lower and 'от' not in lower and numbers:
            result['to'] = float(numbers[0])
        elif 'от' in lower and numbers:
            result['from'] = float(numbers[0])
            if len(numbers) >= 2:
                result['to'] = float(numbers[1])
        elif '—' in salary_text or '–' in salary_text or '-' in salary_text:
            if len(numbers) >= 2:
                result['from'] = float(numbers[0])
                result['to'] = float(numbers[1])
            elif numbers:
                result['from'] = float(numbers[0])
        elif numbers:
            result['from'] = float(numbers[0])

        return result

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Извлекает основной текст описания — листовые div-блоки с длинным текстом."""
        candidates = []
        for elem in soup.find_all(['div', 'section']):
            # Берём только листовые элементы (без вложенных блоков)
            if elem.find(['div', 'section']):
                continue
            text = elem.get_text(strip=True)
            if 100 < len(text) < 6000:
                candidates.append(text)
        return max(candidates, key=len) if candidates else ''

    def _extract_section(self, description: str, markers: list[str]) -> str:
        """Извлекает подраздел описания по маркерам (Обязанности, Требования)."""
        for marker in markers:
            pattern = rf'{marker}[:\s]*(.*?)(?=\n[А-ЯЁ][^\n]{{3,}}:|\Z)'
            m = re.search(pattern, description, re.S | re.I)
            if m:
                return m.group(1).strip()[:2000]
        return ''

    def _extract_date(self, page_text: str) -> str | None:
        now = datetime.now(_MSK)
        # "Сегодня в 12:30"
        m = re.search(r'[Сс]егодня(?:\s+в\s+(\d{1,2}:\d{2}))?', page_text)
        if m:
            if m.group(1):
                h, mn = m.group(1).split(':')
                return now.replace(hour=int(h), minute=int(mn), second=0).strftime('%Y-%m-%d %H:%M:%S')
            return now.strftime('%Y-%m-%d %H:%M:%S')

        # "Вчера"
        if re.search(r'[Вв]чера', page_text):
            return (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

        # "23 апреля" / "23 апреля 2025"
        ru_months = {
            'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
            'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
            'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12,
        }
        months_pat = '|'.join(ru_months)
        m = re.search(rf'(\d{{1,2}})\s+({months_pat})(?:\s+(\d{{4}}))?', page_text, re.I)
        if m:
            day = int(m.group(1))
            month = ru_months[m.group(2).lower()]
            year = int(m.group(3)) if m.group(3) else now.year
            if month > now.month and not m.group(3):
                year -= 1
            try:
                return datetime(year, month, day).strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        return None

    def _map_schedule(self, raw: str) -> str:
        if not raw:
            return ''
        t = re.sub(r'\s+', ' ', raw).lower()
        if 'сменн' in t or '2/2' in t or '3/3' in t:
            return 'Сменный график'
        if 'гибк' in t or 'свободн' in t:
            return 'Гибкий график'
        if 'удал' in t or 'дистанц' in t or 'remote' in t:
            return 'Удаленная работа'
        if 'стандартн' in t or 'полн' in t or '5/2' in t:
            return 'Полный день'
        if 'вахт' in t:
            return 'Вахтовый метод'
        return raw.strip()

    # ── Main loader ───────────────────────────────────────────────────

    def load_vacancies(
        self,
        search_terms: list[str],
        areas: list[str],
        pages: int = 5,
        items_on_page: int = 20,
        delay: int = 2,
        use_progress_bar: bool = True,
    ) -> pd.DataFrame | None:
        area_ids = []
        for area in areas:
            if area in self.SJ_AREAS:
                area_ids.append(self.SJ_AREAS[area])
            else:
                print(f"Регион '{area}' не найден")

        if not area_ids:
            print('Не найдено ни одного валидного региона')
            return None

        all_vacancies = []
        total = len(search_terms) * len(area_ids) * pages

        pbar = tqdm(total=total, desc='SJ Вакансии') if use_progress_bar else None

        try:
            for search_term in search_terms:
                if self._check_404_limit():
                    break
                for area_id in area_ids:
                    if self._check_404_limit():
                        break
                    for page in range(1, pages + 1):
                        if self._check_404_limit():
                            break

                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({
                                'Вакансии': len(all_vacancies),
                                'Запрос': search_term[:20],
                                'Стр.': f'{page}/{pages}',
                            })

                        html = self.search_vacancies(search_term, area_id, page)
                        if html:
                            vacancies = self.parse_vacancies_page(html)
                            if not vacancies:
                                if pbar:
                                    pbar.set_postfix({'Статус': 'Нет вакансий — стоп'})
                                break
                            for v in vacancies:
                                v['search_query'] = search_term
                                v['area_id'] = area_id
                            all_vacancies.extend(vacancies)
                        else:
                            break

                        if delay > 0:
                            time.sleep(delay)

        except StopIteration:
            print('Парсинг прерван: превышен лимит ошибок 404')

        if pbar:
            pbar.close()

        if all_vacancies:
            df = self.create_dataframe(all_vacancies)
            print(f'=== ИТОГИ SJ ===\nВакансий: {len(df)}, уникальных: {df["id"].nunique()}')
            return df
        print('Не найдено ни одной вакансии')
        return None

    # ── DataFrame & Storage ───────────────────────────────────────────

    def create_dataframe(self, vacancies: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(vacancies)
        if df.empty:
            return df

        df['parsed_date'] = _now_msk()

        for col in ['published_at', 'created_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        for col in ['salary_from', 'salary_to']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def save_to_files(self, df: pd.DataFrame, base_filename: str = 'sj_vacancies',
                      vacancy_name: str = '', include_timestamp: bool = True) -> str | None:
        if df is None or df.empty:
            return None
        ts = datetime.now(_MSK).strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename = f'{base_filename}_{vacancy_name}_{ts}.csv' if ts else f'{base_filename}_{vacancy_name}.csv'
        try:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f'Сохранено: {filename}')
            return filename
        except Exception as e:
            print(f'Ошибка сохранения: {e}')
            return None

    def save_to_clickhouse(self, df: pd.DataFrame, clickhouse) -> bool:
        if df is None or df.empty:
            print('Нет данных для сохранения в ClickHouse')
            return False
        try:
            ids = df['id'].tolist()
            clickhouse.execute(f"ALTER TABLE hh_vacancies DELETE WHERE id IN {ids} AND source = 'superjob.ru'")
            clickhouse.insert_dataframe(
                f"INSERT INTO hh_vacancies ({', '.join(df.columns)}) VALUES", df
            )
            print(f'Сохранено {len(df)} вакансий SuperJob в ClickHouse')
            return True
        except Exception as e:
            print(f'Ошибка ClickHouse: {e}')
            return False

    def get_available_areas(self) -> list[str]:
        return list(self.SJ_AREAS.keys())

    def find_area_id(self, area_name: str) -> int | None:
        return self.SJ_AREAS.get(area_name)
