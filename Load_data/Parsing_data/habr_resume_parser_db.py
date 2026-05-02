"""
Парсер резюме с Habr Career (career.habr.com).

Профили публичны и рендерятся на сервере (Vue SSR) — не требует авторизации.
Поиск: career.habr.com/resumes?q=QUERY&page=N
Профиль: career.habr.com/{username}
"""
import requests
import time
import re
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_MSK = ZoneInfo('Europe/Moscow')


def _now_msk() -> str:
    return datetime.now(_MSK).strftime('%Y-%m-%d %H:%M:%S')


# Пути, которые не являются профилями пользователей
_EXCLUDED_PATHS = {
    '/vacancies', '/resumes', '/experts', '/companies', '/salaries',
    '/education', '/journal', '/info', '/catalog', '/sitemap',
    '/feedback', '/price', '/universities', '/education_centers',
    '/users', '/rating', '/ratings',
}
_EXCLUDED_PREFIXES = (
    '/companies/', '/universities/', '/education_centers/', '/vacancies/',
    '/journal/', '/catalog/', '/info/', '/experts/', '/salaries/',
    '/education/', '/users/', '/rating',
)


def _is_profile_link(href: str) -> bool:
    """Возвращает True если href — ссылка на профиль соискателя."""
    if not href or not re.match(r'^/[a-zA-Z][a-zA-Z0-9_-]{2,}$', href):
        return False
    if href in _EXCLUDED_PATHS:
        return False
    return not any(href.startswith(p) for p in _EXCLUDED_PREFIXES)


class HabrResumeParser:
    """Парсер резюме с Habr Career."""

    def __init__(self, timeout: int = 30, max_retries: int = 3, max_404_errors: int = 10):
        self.base_url = 'https://career.habr.com'
        self.search_url = 'https://career.habr.com/resumes'
        self.timeout = timeout
        self.consecutive_404_errors = 0
        self.max_404_errors = max_404_errors

        retry = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],  # 429 обрабатываем вручную
            allowed_methods=['GET'],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://career.habr.com/resumes',
        })

    # ── 404 handling ─────────────────────────────────────────────────

    def _check_404_limit(self) -> bool:
        return self.consecutive_404_errors >= self.max_404_errors

    def _handle_404(self):
        self.consecutive_404_errors += 1
        if self._check_404_limit():
            raise StopIteration('Превышен лимит 404 ошибок')

    def _handle_ok(self):
        if self.consecutive_404_errors > 0:
            self.consecutive_404_errors = 0

    # ── Search page ───────────────────────────────────────────────────

    def search_resumes(self, keywords: str, page: int = 1) -> str | None:
        """Возвращает HTML страницы поиска или None."""
        if self._check_404_limit():
            return None
        params = {'q': keywords}
        if page > 1:
            params['page'] = page
        try:
            r = self.session.get(self.search_url, params=params, timeout=self.timeout)
            if r.status_code == 200:
                self._handle_ok()
                return r.text
            elif r.status_code == 404:
                self._handle_404()
            else:
                self._handle_ok()
        except StopIteration:
            raise
        except Exception:
            self._handle_ok()
        return None

    def parse_search_results(self, html: str) -> list[dict]:
        """
        Извлекает резюме прямо из карточек на странице поиска.
        Не посещает отдельные профили — исключает 429 Rate Limit.
        """
        if not html:
            return []
        soup = BeautifulSoup(html, 'html.parser')
        resumes, seen = [], set()

        for article in soup.find_all('article'):
            # Находим ссылку на профиль внутри карточки
            profile_link = next(
                (a for a in article.find_all('a', href=True)
                 if _is_profile_link(a.get('href', ''))),
                None,
            )
            if profile_link is None:
                continue
            href = profile_link.get('href')
            if href in seen:
                continue
            seen.add(href)

            username = href.lstrip('/')
            url = urljoin(self.base_url, href)
            name = profile_link.get_text(strip=True)

            resume = self._parse_card(username, url, name, article)
            resumes.append(resume)

        return resumes

    def _parse_card(self, username: str, url: str, name: str, card) -> dict:
        """Разбирает карточку резюме из страницы поиска."""
        # Полный текст карточки строками
        lines = [l.strip() for l in card.get_text(separator='\n').split('\n') if l.strip()]

        # Дата последнего обновления
        t_elem = card.find('time')
        last_update = t_elem.get_text(strip=True) if t_elem else ''

        # Навыки — кнопки внутри карточки (исключаем UI-кнопки и склеенный текст)
        _UI_BUTTONS = {'Развернуть', 'Свернуть', 'Открыть контакты', 'Написать', 'Откликнуться'}
        skills = [
            b.get_text(strip=True) for b in card.find_all('button')
            if b.get_text(strip=True)
            and b.get_text(strip=True) not in _UI_BUTTONS
            and '•' not in b.get_text()
            and 'компани' not in b.get_text().lower()
            and len(b.get_text(strip=True)) < 60
        ]

        # Уровень
        level = self._extract_level('\n'.join(lines))

        # Заголовок — текст между именем и уровнем/статусом
        title = self._extract_title_from_lines(lines, name, level)

        # Зарплата
        salary = self._extract_salary('\n'.join(lines))

        # Статус поиска
        applicant_status = self._extract_status('\n'.join(lines))

        # Возраст
        age = self._extract_age_from_lines(lines)

        # Опыт
        total_exp = self._extract_exp_from_lines(lines)
        experience_months = self._parse_experience_to_months(total_exp)

        # Сниппет описания опыта — текст карточки после «Опыт работы»
        exp_description = self._extract_exp_description_from_lines(lines)

        return {
            'id':                            username,
            'title':                         f'{title} — {level}'.strip(' —') if level else title,
            'url':                           url,
            'specialization':                [title] if title else [],
            'last_company':                  '',
            'last_position':                 '',
            'last_experience_description':   exp_description,
            'last_company_experience_period': '',  # недоступно из листинга
            'skills':                        skills,
            'education':                     [],
            'courses':                       [],
            'salary':                        salary,
            'age':                           age,
            'total_experience':              total_exp,
            'experience_months':             experience_months,
            'location':                      '',
            'gender':                        '',
            'applicant_status':              applicant_status,
            'source':                        'career.habr.com',
        }

    def _extract_title_from_lines(self, lines: list[str], name: str, level: str) -> str:
        """Должность — строки между именем и уровнем/статусом в карточке."""
        _level_re = re.compile(r'^(Lead|Senior|Middle|Junior|Ведущий|Старший|Средний|Младший)$')
        _stop = {'Ищу работу', 'Рассматриваю предложения', 'Не ищу работу',
                 'Профессиональные навыки', 'Возраст', 'Опыт работы',
                 'Написать на Хабре', 'Написать', 'Войти', '•', ''}
        try:
            start = next(i for i, l in enumerate(lines) if l == name)
        except StopIteration:
            return ''
        title_parts = []
        for l in lines[start + 2: start + 8]:   # +2 пропускаем дату
            if l in _stop or _level_re.match(l) or '₽' in l:
                break
            if l != '•' and len(l) > 1:
                title_parts.append(l)
        return ' '.join(title_parts).strip()

    def _extract_age_from_lines(self, lines: list[str]) -> int | None:
        """Возраст — строка после «Возраст»."""
        try:
            idx = next(i for i, l in enumerate(lines) if l == 'Возраст')
            m = re.search(r'(\d+)', lines[idx + 1])
            return int(m.group(1)) if m else None
        except (StopIteration, IndexError):
            return None

    def _extract_exp_from_lines(self, lines: list[str]) -> str:
        """Опыт работы — строка после «Опыт работы»."""
        try:
            idx = next(i for i, l in enumerate(lines) if l == 'Опыт работы')
            for l in lines[idx + 1: idx + 5]:
                if re.search(r'лет|год|месяц', l, re.I):
                    return l
        except StopIteration:
            pass
        return ''

    def _extract_exp_description_from_lines(self, lines: list[str]) -> str:
        """
        Сниппет описания опыта — текст карточки после блока «Опыт работы».
        Listing показывает краткое описание последней позиции без названия компании.
        """
        _STOP = {'Образование', 'Профессиональные навыки', 'Добавить в избранное',
                 'Открыть контакты', 'Написать', 'Хабр Карьера'}
        try:
            idx = next(i for i, l in enumerate(lines) if l == 'Опыт работы')
        except StopIteration:
            return ''

        # Пропускаем строки с количеством компаний, стажем и разделителями
        desc_parts = []
        skip_re = re.compile(r'^\d+\s+компани|\d+\s*(лет|год|года|месяц)|^•$', re.I)
        for l in lines[idx + 1: idx + 30]:
            if l in _STOP:
                break
            if skip_re.search(l):   # пропускаем строки со стажем на любом этапе
                continue
            if len(l) > 5:
                desc_parts.append(l)
            if len(' '.join(desc_parts)) > 500:
                break

        return ' '.join(desc_parts).strip()[:1000]

    # ── Profile page ──────────────────────────────────────────────────

    def parse_resume_details(self, url: str, max_retries: int = 2) -> dict | None:
        """
        Разбирает профиль соискателя.
        Возвращает dict — успех, {} — 404/ошибка, None — 429 (rate limit).
        """
        if self._check_404_limit():
            return {}
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    self._handle_ok()
                    return self._extract_profile(url, r.text)
                elif r.status_code == 404:
                    self._handle_404()
                    return {}
                elif r.status_code == 429:
                    # Сигнализируем вызывающему коду о rate limit, не ждём здесь
                    return None
                else:
                    self._handle_ok()
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            except StopIteration:
                raise
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return {}

    def _extract_profile(self, url: str, html: str) -> dict:
        soup = BeautifulSoup(html, 'html.parser')
        page_text = soup.get_text(separator='\n', strip=True)
        lines = [l.strip() for l in page_text.split('\n') if l.strip()]

        # username / id
        username = url.rstrip('/').split('/')[-1]

        # Имя — h1
        h1 = soup.find('h1')
        name = h1.get_text(strip=True) if h1 else username

        # Заголовок (title) — строки между h1 и "Возраст:"
        title = self._extract_title(lines, name)

        # Уровень: Junior/Middle/Senior/Lead
        level = self._extract_level(page_text)

        # Зарплата
        salary = self._extract_salary(page_text)

        # Статус поиска
        applicant_status = self._extract_status(page_text)

        # Демографические данные
        age = self._extract_age(page_text)
        total_experience = self._extract_field(
            page_text, r'Опыт работы:\s*\n+([^\n]{3,50})'
        )
        experience_months = self._parse_experience_to_months(total_experience)
        location = self._extract_field(page_text, r'Местоположение:\s*\n+([^\n]{3,80})')

        # Навыки — секция "Навыки"
        skills = self._extract_skills(lines)

        # Опыт работы — первое место
        last_company, last_position, last_period, last_description = \
            self._extract_last_experience(lines)

        # Образование
        education = self._extract_education(lines)

        # Обо мне — секция между "Обо мне" и "Навыки" / "Опыт работы"
        about = self._extract_about(lines)

        return {
            'id':                            username,
            'title':                         f'{title} — {level}'.strip(' —') if level else title,
            'url':                           url,
            'specialization':                [title] if title else [],
            'last_company':                  last_company,
            'last_position':                 last_position,
            'last_experience_description':   last_description,
            'last_company_experience_period': last_period,
            'skills':                        skills,
            'education':                     [education] if education else [],
            'courses':                       [],
            'salary':                        salary,
            'age':                           age,
            'total_experience':              total_experience,
            'experience_months':             experience_months,
            'location':                      location,
            'gender':                        '',
            'applicant_status':              applicant_status,
            'source':                        'career.habr.com',
        }

    # ── Field extractors ──────────────────────────────────────────────

    def _extract_field(self, text: str, pattern: str, group: int = 1) -> str:
        m = re.search(pattern, text, re.I | re.M)
        return m.group(group).strip() if m else ''

    def _extract_title(self, lines: list[str], name: str) -> str:
        """Извлекает профессиональный заголовок профиля (строка перед уровнем)."""
        try:
            name_idx = next(i for i, l in enumerate(lines) if l == name)
        except StopIteration:
            return ''

        _level_re = re.compile(
            r'(Ведущий|Старший|Средний|Младший|Junior|Middle|Senior|Lead)\s*[\(\[]?'
        )
        _bio_re = re.compile(r'https?:|t\.me|@\w{4,}|telegram|github|linkedin', re.I)

        # Ищем индекс уровня
        level_idx = None
        for i in range(name_idx + 1, min(name_idx + 12, len(lines))):
            if _level_re.search(lines[i]):
                level_idx = i
                break

        # Перебираем строки между именем и уровнем — берём первую «чистую» профессиональную
        upper = level_idx if level_idx else name_idx + 8
        for l in lines[name_idx + 1: upper]:
            # Пропускаем bio с url/handles или слишком длинные
            if _bio_re.search(l) or len(l) > 80:
                continue
            # Пропускаем служебные строки
            if l in ('Ищу работу', 'Рассматриваю предложения', 'Не ищу работу',
                     'Написать', 'Войти', 'Контакты', 'Поднимите резюме'):
                continue
            if len(l) > 3:
                return l
        return ''

    def _extract_level(self, text: str) -> str:
        m = re.search(r'(Ведущий|Старший|Средний|Младший)\s*\(?\s*(?:Lead|Senior|Middle|Junior)?\s*\)?', text, re.I)
        if not m:
            m = re.search(r'\b(Lead|Senior|Middle|Junior)\b', text, re.I)
        return m.group(0).strip() if m else ''

    def _extract_salary(self, text: str) -> str:
        m = re.search(r'((?:от|до)\s*[\d\s]+₽|[\d\s]{5,}₽)', text, re.I)
        return m.group(1).strip() if m else ''

    def _extract_status(self, text: str) -> str:
        for status in ('Ищу работу', 'Рассматриваю предложения', 'Не ищу работу'):
            if status in text:
                return status
        return ''

    def _extract_age(self, text: str) -> int | None:
        m = re.search(r'Возраст:\s*\n+(\d+)\s+(?:лет|год|года)', text)
        return int(m.group(1)) if m else None

    def _parse_experience_to_months(self, experience_text: str) -> int | None:
        if not experience_text:
            return None
        months = 0
        y = re.search(r'(\d+)\s*(?:год|лет)', experience_text)
        if y:
            months += int(y.group(1)) * 12
        mo = re.search(r'(\d+)\s*месяц', experience_text)
        if mo:
            months += int(mo.group(1))
        return months if months > 0 else None

    def _extract_skills(self, lines: list[str]) -> list[str]:
        """Извлекает навыки из секции 'Навыки'."""
        try:
            start = next(i for i, l in enumerate(lines) if l == 'Навыки')
        except StopIteration:
            return []
        skills = []
        for l in lines[start + 1:start + 60]:
            if l in ('Опыт работы', 'Образование', 'Обо мне', 'Рекомендательные письма', 'Друзья'):
                break
            if re.match(r'^[A-Za-zА-Яа-яёЁ0-9.#+\-\s/]{2,40}$', l) and l not in (
                'Выберите навык', 'чтобы посмотреть'
            ):
                skills.append(l)
        return skills

    def _extract_about(self, lines: list[str]) -> str:
        """Извлекает раздел 'Обо мне'."""
        try:
            start = next(i for i, l in enumerate(lines) if l == 'Обо мне')
        except StopIteration:
            return ''
        parts = []
        for l in lines[start + 1:start + 40]:
            if l in ('Навыки', 'Опыт работы', 'Образование', 'Рекомендательные письма'):
                break
            parts.append(l)
        return ' '.join(parts)[:2000]

    def _extract_last_experience(self, lines: list[str]) -> tuple[str, str, str, str]:
        """Возвращает (company, position, period, description) первого места работы."""
        try:
            start = next(i for i, l in enumerate(lines) if l == 'Опыт работы')
        except StopIteration:
            return '', '', '', ''

        company, position, period, desc_lines = '', '', '', []
        # Паттерн периода: "Март 2022 — Апрель 2025  (3 года ...)"
        period_pat = re.compile(r'[А-Я][а-я]+\s+\d{4}\s*[—–-]\s*(?:[А-Я][а-я]+\s+\d{4}|[Нн]астоящее)')
        state = 'company'

        end_markers = {'Образование', 'Навыки', 'Обо мне', 'Рекомендательные письма',
                       'Хабр Карьера', 'О сервисе'}

        # Паттерн: одиночное слово с заглавной буквы → скорее всего город
        _city_re = re.compile(r'^[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?$')

        for l in lines[start + 1: start + 100]:
            if l in end_markers:
                break
            if not company:
                company = l
                continue
            if not position and not period_pat.match(l):
                # Пропускаем строки, похожие на город (одно слово с большой буквы)
                if _city_re.match(l):
                    continue
                if len(l) > 3:
                    position = l
                continue
            if period_pat.match(l):
                period = l
                state = 'desc'
                continue
            if state == 'desc':
                desc_lines.append(l)

        description = ' '.join(desc_lines)[:1500]
        return company, position, period, description

    # Известные строки-артефакты, которые не являются названиями вузов
    _EDU_NOISE = frozenset({
        'Хабр Карьера', 'О сервисе', 'Услуги и цены', 'Журнал', 'Каталог вакансий',
        'Каталог курсов', 'Карта сайта', 'Контакты', 'Отзывы об онлайн-школах',
        'Рейтинг школ', 'Промокоды и скидки', 'Для соискателя', 'Для работодателя',
        'API сервиса', 'Служба поддержки', 'Следите за нами в соцсетях',
    })

    def _extract_education(self, lines: list[str]) -> str:
        """Извлекает название учебного заведения."""
        try:
            start = next(i for i, l in enumerate(lines) if l == 'Образование')
        except StopIteration:
            return ''
        # Первая строка, похожая на название вуза (не навигационный шум)
        for l in lines[start + 1: start + 6]:
            if l in self._EDU_NOISE or len(l) < 3:
                continue
            # Вузы обычно не содержат / и не начинаются с маленькой буквы
            if not l.startswith('/') and not l[0].islower():
                return l
        return ''

    # ── Main loader ───────────────────────────────────────────────────

    # Порог 429 подряд — после него пропускаем поисковый запрос и ждём
    _MAX_CONSECUTIVE_429 = 3
    _429_COOLDOWN = 120  # секунд ожидания после достижения порога

    def load_resumes(
        self,
        search_terms: list[str],
        pages: int = 5,
        items_on_page: int = 20,
        delay: int = 5,
        use_progress_bar: bool = True,
    ) -> pd.DataFrame | None:

        total = len(search_terms) * pages
        pbar = tqdm(total=total, desc='Habr Резюме') if use_progress_bar else None
        all_resumes = []
        consecutive_429 = 0

        try:
            for search_term in search_terms:
                if self._check_404_limit():
                    break
                for page in range(1, pages + 1):
                    if self._check_404_limit():
                        break
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({'Резюме': len(all_resumes), 'Стр.': f'{page}/{pages}'})

                    html = self.search_resumes(search_term, page)
                    if not html:
                        break

                    # Данные извлекаются прямо из карточек — профили не посещаем
                    resumes = self.parse_search_results(html)
                    if not resumes:
                        if pbar:
                            pbar.set_postfix({'Статус': 'Нет карточек — стоп'})
                        break

                    for resume in resumes:
                        resume['search_query'] = search_term
                    all_resumes.extend(resumes)

                    if delay > 0:
                        time.sleep(delay)

        except StopIteration:
            print('Парсинг прерван: превышен лимит 404')

        if pbar:
            pbar.close()

        if all_resumes:
            df = self.create_dataframe(all_resumes)
            print(f'=== ИТОГИ Habr ===\nРезюме: {len(df)}, уникальных id: {df["id"].nunique()}')
            return df
        print('Не найдено ни одного резюме')
        return None

    # ── DataFrame & storage ───────────────────────────────────────────

    def create_dataframe(self, resumes: list[dict]) -> pd.DataFrame:
        simplified = []
        for r in resumes:
            simplified.append({
                'id':                            r.get('id', ''),
                'title':                         r.get('title', ''),
                'url':                           r.get('url', ''),
                'specialization':                r.get('specialization', []),
                'last_company':                  r.get('last_company', ''),
                'last_position':                 r.get('last_position', ''),
                'last_experience_description':   r.get('last_experience_description', ''),
                'last_company_experience_period': r.get('last_company_experience_period', ''),
                'skills':                        r.get('skills', []),
                'education':                     r.get('education', []),
                'courses':                       r.get('courses', []),
                'salary':                        r.get('salary', ''),
                'age':                           r.get('age'),
                'total_experience':              r.get('total_experience', ''),
                'experience_months':             r.get('experience_months'),
                'location':                      r.get('location', ''),
                'gender':                        r.get('gender', ''),
                'applicant_status':              r.get('applicant_status', ''),
                'search_query':                  r.get('search_query', ''),
                'source':                        'career.habr.com',
                'parsed_date':                   _now_msk(),
            })
        return pd.DataFrame(simplified)

    def save_to_files(self, df: pd.DataFrame, base_filename: str = 'habr_resumes',
                      resume_name: str = '', include_timestamp: bool = True) -> str | None:
        if df is None or df.empty:
            return None
        ts = datetime.now(_MSK).strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename = f'{base_filename}_{resume_name}_{ts}.csv'.replace('__', '_')
        try:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f'Сохранено: {filename}')
            return filename
        except Exception as e:
            print(f'Ошибка: {e}')
            return None

    def save_to_clickhouse(self, df: pd.DataFrame, clickhouse) -> bool:
        if df is None or df.empty:
            print('Нет данных для ClickHouse')
            return False
        try:
            ids = df['id'].tolist()
            clickhouse.execute(f"ALTER TABLE hh_resumes DELETE WHERE id IN {ids} AND source = 'career.habr.com'")
            clickhouse.insert_dataframe(
                f"INSERT INTO hh_resumes ({', '.join(df.columns)}) VALUES", df
            )
            print(f'Сохранено {len(df)} резюме Habr в ClickHouse')
            return True
        except Exception as e:
            print(f'Ошибка ClickHouse: {e}')
            return False
