import requests
import time
import json
import re
import pandas as pd
import numpy as np
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
from clickhouse_driver import Client as ClickhouseClient

class HHVacancyParser:
    """
    Парсер вакансий с сайта hh.ru (HTML-скрапинг)
    """
    
    # Словарь регионов HH
    HH_AREAS = {
        "Москва": 1,
        "Санкт-Петербург": 2,
        "Екатеринбург": 3,
        "Новосибирск": 4,
        "Нижний Новгород": 66,
        "Казань": 68,
        "Самара": 72,
        "Омск": 76,
        "Челябинск": 78,
        "Красноярск": 88,
        "Пермь": 99,
        "Воронеж": 104,
        "Уфа": 1118,
        "Ростов-на-Дону": 113,
        "Краснодар": 1146,
        "Волгоград": 115,
        "Саратов": 1199,
        "Тюмень": 1202,
        "Тольятти": 1214,
        "Ижевск": 1229,
        "Барнаул": 1255,
        "Владивосток": 1261,
        "Иркутск": 1299,
        "Калининград": 1307,
        "Новокузнецк": 1315,
        "Хабаровск": 1347,
        "Ярославль": 1374,
        "Махачкала": 1384,
        "Томск": 1398,
        "Оренбург": 1416,
        "Кемерово": 1438,
        "Рязань": 1450,
        "Астрахань": 1468,
        "Пенза": 1475,
        "Набережные Челны": 1495,
        "Липецк": 1515,
        "Киров": 1523,
        "Чебоксары": 1530,
        "Тула": 1533,
        "Курск": 1542,
        "Улан-Удэ": 1548,
        "Ставрополь": 1564,
        "Сочи": 1574,
        "Тверь": 1586,
        "Магнитогорск": 1598,
        "Иваново": 1604,
        "Брянск": 1624,
        "Белгород": 1635,
        "Сургут": 1646,
        "Владимир": 1665,
        "Чита": 1679,
        "Смоленск": 1694,
        "Калуга": 1708,
        "Орёл": 1723,
        "Волжский": 1736,
        "Череповец": 1759,
        "Саранск": 1771,
        "Вологда": 1785,
        "Владикавказ": 1804,
        "Мурманск": 1817,
        "Якутск": 1837,
        "Грозный": 1846,
        "Таганрог": 1862,
        "Стерлитамак": 1874,
        "Кострома": 1880,
        "Петрозаводск": 1889,
        "Нижневартовск": 1900,
        "Йошкар-Ола": 1913,
        "Новороссийск": 1929,
        "Балашиха": 1946,
        "Химки": 1955,
        "Подольск": 1960,
        "Королёв": 1966,
        "Мытищи": 1975,
        "Люберцы": 1987,
        "Электросталь": 2019,
        "Красногорск": 2038,
        "Коломна": 2052,
        "Одинцово": 2062,
        "Домодедово": 2074,
        "Реутов": 2086,
        "Серпухов": 2104,
        "Раменское": 2128,
        "Пушкино": 2159,
        "Воскресенск": 2175,
        "Долгопрудный": 2191,
        "Жуковский": 2209,
        "Украина": 5,
        "Беларусь": 9,
        "Казахстан": 10,
        "Азербайджан": 11,
        "Кыргызстан": 12,
        "Грузия": 97,
        "Армения": 98,
        "Узбекистан": 1237,
        "Молдова": 1239,
        "Туркменистан": 1243,
        "Таджикистан": 1245,
        "Германия": 10619,
        "Великобритания": 10620,
        "Франция": 10621,
        "Италия": 10622,
        "Испания": 10623,
        "Польша": 10624,
        "Нидерланды": 10625,
        "Бельгия": 10626,
        "Чехия": 10627,
        "Швейцария": 10628,
        "Австрия": 10629,
        "Швеция": 10630,
        "Норвегия": 10631,
        "Финляндия": 10632,
        "Дания": 10633,
        "Португалия": 10634,
        "Ирландия": 10635,
        "Венгрия": 10636,
        "Румыния": 10637,
        "Греция": 10638,
        "Болгария": 10639,
        "Сербия": 10640,
        "Хорватия": 10641,
        "Словакия": 10642,
        "Словения": 10643,
        "США": 10644,
        "Канада": 10645,
        "Бразилия": 10646,
        "Мексика": 10647,
        "Аргентина": 10648,
        "Китай": 10649,
        "Япония": 10650,
        "Южная Корея": 10651,
        "Сингапур": 10652,
        "Индия": 10653,
        "ОАЭ": 10654,
        "Турция": 10655,
        "Израиль": 10656,
        "Таиланд": 10657,
        "Вьетнам": 10658,
        "Малайзия": 10659,
        "Индонезия": 10660,
        "Филиппины": 10661,
        "Австралия": 10662,
        "Новая Зеландия": 10663,
        "ЮАР": 10664,
        "Все регионы": 0,
        "Города с населением более 1 млн": 1001,
        "Города с населением 500 тыс. - 1 млн": 1002,
        "Регионы России (кроме Москвы и СПб)": 1017,
        "Другие регионы": 1024,
        "Удаленная работа": 1036,
        "Россия": 113
    }

    def __init__(self, timeout=30, max_retries=3, max_404_errors=5):
        self.base_url = 'https://hh.ru'
        self.search_url = 'https://hh.ru/search/vacancy'
        self.timeout = timeout
        self.session = requests.Session()
        self.consecutive_404_errors = 0
        self.max_404_errors = max_404_errors

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://hh.ru/search/vacancy',
        })
    
    def _check_404_limit(self):
        if self.consecutive_404_errors >= self.max_404_errors:
            return True
        return False
    
    def _handle_404_error(self):
        self.consecutive_404_errors += 1
        if self._check_404_limit():
            raise StopIteration("Превышен лимит ошибок 404")
    
    def _handle_successful_request(self):
        if self.consecutive_404_errors > 0:
            self.consecutive_404_errors = 0

    def search_vacancies(self, search_text='Аналитик данных', area=1, page=0, per_page=20):
        if self._check_404_limit():
            return None

        params = {
            'text': search_text,
            'area': area,
            'page': page,
            'per_page': per_page,
        }

        try:
            response = self.session.get(
                self.search_url,
                params=params,
                timeout=self.timeout,
                allow_redirects=True
            )

            if response.status_code == 200:
                self._handle_successful_request()
                return response.text
            elif response.status_code == 404:
                self._handle_404_error()
                return None
            else:
                self._handle_successful_request()
                print(f"[{response.status_code}] {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            self._handle_successful_request()
            return None
        except requests.exceptions.ConnectionError:
            self._handle_successful_request()
            return None
        except StopIteration as e:
            raise e
        except Exception:
            self._handle_successful_request()
            return None

    def parse_vacancies_page(self, html):
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        vacancies = []

        vacancy_elements = soup.find_all(
            ['div', 'article'],
            attrs={'data-qa': re.compile(r'^vacancy-serp__vacancy')}
        )

        for element in vacancy_elements:
            vacancy = self._parse_vacancy_card(element)
            if vacancy:
                vacancies.append(vacancy)

        return vacancies

    def _parse_vacancy_card(self, element):
        try:
            vacancy = {
                'salary_from': None, 'salary_to': None,
                'salary_currency': None, 'salary_gross': None,
                'requirement': '', 'responsibility': '',
                'experience': '', 'employment': '', 'schedule': '',
                'area': '', 'employer': '',
                'published_at': None, 'created_at': None,
                'description': '',
            }

            title_link = element.find('a', {'data-qa': 'serp-item__title'})
            if not title_link:
                return None

            href = title_link.get('href', '')
            match = re.search(r'/vacancy/(\d+)', href)
            if not match:
                return None

            vacancy['id'] = match.group(1)
            vacancy['url'] = f"{self.base_url}/vacancy/{vacancy['id']}"
            vacancy['alternate_url'] = vacancy['url']
            vacancy['name'] = title_link.get_text(strip=True)

            employer_elem = (
                element.find('a', {'data-qa': 'vacancy-serp__vacancy-employer'}) or
                element.find('span', {'data-qa': 'vacancy-serp__vacancy-employer'})
            )
            if employer_elem:
                vacancy['employer'] = employer_elem.get_text(strip=True)

            # area — span, не div (Magritte)
            area_elem = element.find('span', {'data-qa': 'vacancy-serp__vacancy-address'})
            if area_elem:
                vacancy['area'] = area_elem.get_text(strip=True)

            # experience — динамический суффикс data-qa
            exp_elem = element.find('span', attrs={'data-qa': re.compile(r'^vacancy-serp__vacancy-work-experience')})
            if exp_elem:
                vacancy['experience'] = exp_elem.get_text(strip=True)

            # salary, employment, schedule, description, published_at — со страницы вакансии
            details = self.get_vacancy_details(vacancy['id'], vacancy['url'])
            if details:
                vacancy.update(details)

            time.sleep(1)
            return vacancy

        except Exception:
            return None

    def _parse_salary(self, salary_text):
        result = {'salary_from': None, 'salary_to': None, 'salary_currency': None, 'salary_gross': None}
        if not salary_text:
            return result

        # Определяем валюту
        if '₽' in salary_text or 'руб' in salary_text.lower():
            result['salary_currency'] = 'RUR'
        elif '$' in salary_text or 'USD' in salary_text:
            result['salary_currency'] = 'USD'
        elif '€' in salary_text or 'EUR' in salary_text:
            result['salary_currency'] = 'EUR'

        # Извлекаем числа (убираем пробелы внутри чисел)
        numbers = [int(re.sub(r'\s', '', m)) for m in re.findall(r'\d[\d\s]*\d|\d', salary_text)
                   if re.sub(r'\s', '', m).isdigit()]

        lower = salary_text.lower()
        if 'от' in lower and 'до' in lower and len(numbers) >= 2:
            result['salary_from'] = float(numbers[0])
            result['salary_to'] = float(numbers[1])
        elif 'от' in lower and len(numbers) >= 1:
            result['salary_from'] = float(numbers[0])
        elif 'до' in lower and len(numbers) >= 1:
            result['salary_to'] = float(numbers[0])
        elif len(numbers) >= 1:
            result['salary_from'] = float(numbers[0])

        return result
    
    def _map_schedule(self, raw):
        """Маппинг Magritte-значений формата работы в исходные категории HH.ru."""
        if not raw:
            return ''
        # Нормализуем все виды пробелов (включая  ) перед сравнением
        t = re.sub(r'\s+', ' ', raw).lower()
        if 'сменн' in t:
            return 'Сменный график'
        if 'гибрид' in t or ('удал' in t and ('или' in t or ', ' in t)):
            return 'Гибкий график'
        if 'удал' in t:
            return 'Удаленная работа'
        if 'на месте' in t or 'офис' in t:
            return 'Полный день'
        return raw

    def _clean_html_tags(self, text):
        if not text:
            return text
        return re.sub(r'<[^>]+>', '', text)
    
    def load_vacancies(self, search_terms, areas, pages=1, items_on_page=100, delay=1, 
                      use_progress_bar=True, stop_on_rate_limit=True):
        area_ids = []
        invalid_areas = []
        
        for area in areas:
            if area in self.HH_AREAS:
                area_ids.append(self.HH_AREAS[area])
            else:
                invalid_areas.append(area)
        
        if invalid_areas:
            print(f"Невалидные регионы: {invalid_areas}")
        
        if not area_ids:
            print("Не найдено ни одного валидного региона")
            return None
            
        all_vacancies = []
        total_requests = len(search_terms) * len(area_ids) * pages
        
        if use_progress_bar:
            pbar = tqdm(total=total_requests, desc="Сбор вакансий")
        
        request_count = 0
        error_count = 0
        
        try:
            for search_term in search_terms:
                if self._check_404_limit():
                    break
                    
                for area_id in area_ids:
                    if self._check_404_limit():
                        break
                        
                    for page in range(pages):
                        if self._check_404_limit():
                            break
                            
                        request_count += 1
                        
                        if use_progress_bar:
                            pbar.update(1)
                            pbar.set_postfix({
                                'Вакансии': len(all_vacancies),
                                'Запрос': search_term[:20],
                                'Страница': f"{page+1}/{pages}"
                            })
                        
                        html = self.search_vacancies(
                            search_text=search_term,
                            area=area_id,
                            page=page,
                            per_page=items_on_page
                        )

                        if html:
                            vacancies = self.parse_vacancies_page(html)

                            for vacancy in vacancies:
                                vacancy['search_query'] = search_term
                                vacancy['area_id'] = area_id

                            all_vacancies.extend(vacancies)

                            if not vacancies:
                                if use_progress_bar:
                                    pbar.set_postfix({
                                        'Вакансии': len(all_vacancies),
                                        'Запрос': search_term[:20],
                                        'Статус': 'Нет вакансий - остановка'
                                    })
                                break
                        else:
                            error_count += 1

                        if delay > 0:
                            time.sleep(delay)
        
        except StopIteration:
            print("Парсинг прерван из-за превышения лимита ошибок 404")
        
        if use_progress_bar:
            pbar.close()
        
        print(f"Сбор завершен. Запросов: {request_count}, Ошибок: {error_count}")
        
        if all_vacancies:
            df = self.create_dataframe(all_vacancies)
            
            print(f"=== ИТОГИ ===")
            print(f"Всего собрано вакансий: {len(df)}")
            print(f"Уникальных вакансий: {df['id'].nunique()}")
            
            if 'search_query' in df.columns:
                print("Распределение по поисковым запросам:")
                for query, count in df['search_query'].value_counts().items():
                    print(f"  {query}: {count}")
            
            return df
        else:
            print("Не найдено ни одной вакансии")
            return None
    
    def _get_area_name(self, area_id):
        for name, id_ in self.HH_AREAS.items():
            if id_ == area_id:
                return name
        return f"Unknown ({area_id})"
    
    def create_dataframe(self, vacancies):
        df = pd.DataFrame(vacancies)
        
        if df.empty:
            return df
        
        df['source'] = 'hh.ru'
        df['parsed_date'] = _now_msk()
        
        date_columns = ['published_at', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                df[col] = df[col].dt.tz_convert(None)
        
        salary_columns = ['salary_from', 'salary_to']
        for col in salary_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def save_to_files(self, df, base_filename='vacancies', vacancy_name='', include_timestamp=True):
        if df is None or df.empty:
            return None
        
        timestamp = datetime.now(_MSK).strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename_suffix = f'_{timestamp}' if timestamp else ''
        
        try:
            df_save = df.copy()
            
            datetime_columns = df_save.select_dtypes(include=['datetime64[ns]']).columns
            for col in datetime_columns:
                df_save[col] = df_save[col].dt.tz_localize(None)
            
            csv_filename = f'{base_filename}_{vacancy_name}{filename_suffix}.csv'
            df_save.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"Данные сохранены в CSV: {csv_filename}")
            
            return csv_filename
            
        except Exception as e:
            print(f"Ошибка при сохранении файлов: {e}")
            try:
                csv_filename = f'{base_filename}{filename_suffix}_backup.csv'
                df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                print(f"Резервная копия сохранена в CSV: {csv_filename}")
                return csv_filename
            except Exception:
                return None
    
    def save_to_clickhouse(self, df, clickhouse):
        """
        Сохранение DataFrame в ClickHouse
        """
        if df is None or df.empty:
            print("Нет данных для сохранения в ClickHouse")
            return False
            
        try:
            for i in range(0, len(df), 10000):
                ids = df.loc[i: i + 10000, 'id'].tolist()
                clickhouse.execute(f"""
                alter table hh_vacancies
                delete where id in {ids}
                """)
            
            clickhouse.insert_dataframe(f"""
                INSERT INTO hh_vacancies
                ({', '.join(df.columns)})
                VALUES
            """, df)
            
            print(f"Успешно сохранено {len(df)} записей в ClickHouse таблицу hh_vacancies")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении в ClickHouse: {e}")
            return False
    
    def get_vacancy_details(self, vacancy_id, url, max_retries=2):
        details = {}

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Описание
                    desc_elem = soup.find('div', {'data-qa': 'vacancy-description'})
                    if desc_elem:
                        details['description'] = desc_elem.get_text(strip=True)

                    # Опыт работы
                    exp_elem = soup.find('span', {'data-qa': 'vacancy-experience'})
                    if exp_elem:
                        details['experience'] = exp_elem.get_text(strip=True)

                    # Тип занятости: "Полная занятость"
                    emp_elem = soup.find('div', {'data-qa': 'common-employment-text'})
                    if emp_elem:
                        details['employment'] = emp_elem.get_text(strip=True)

                    # Формат работы → маппинг в исходные категории HH.ru
                    sched_elem = soup.find('p', {'data-qa': 'work-formats-text'})
                    if sched_elem:
                        raw = re.sub(r'^Формат работы:\s*', '', sched_elem.get_text(strip=True))
                        details['schedule'] = self._map_schedule(raw)

                    # Зарплата: "165 000₽за месяцна руки"
                    salary_elem = soup.find('div', {'data-qa': 'vacancy-salary'})
                    if salary_elem:
                        salary_text = salary_elem.get_text(strip=True)
                        details.update(self._parse_salary(salary_text))
                        # Gross: "до вычета налогов" → True, "на руки" → False
                        if 'до вычета' in salary_text.lower() or 'gross' in salary_text.lower():
                            details['salary_gross'] = True
                        elif 'на руки' in salary_text.lower():
                            details['salary_gross'] = False

                    # Дата публикации: "опубликована1 мая 2026" (без пробела в HTML)
                    page_text = soup.get_text()
                    date_match = re.search(
                        r'опубликована\s*(\d{1,2})\s*(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*(\d{4})',
                        page_text, re.IGNORECASE
                    )
                    if date_match:
                        ru_months = {
                            'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
                            'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
                            'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
                        }
                        day, month_ru, year = date_match.group(1), date_match.group(2).lower(), date_match.group(3)
                        month_num = ru_months.get(month_ru, '01')
                        iso_date = f"{year}-{month_num}-{int(day):02d}"
                        details['published_at'] = iso_date
                        details['created_at'] = iso_date

                    return details
                elif response.status_code == 404:
                    return {}
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        return {}
    
    def get_available_areas(self):
        return list(self.HH_AREAS.keys())
    
    def find_area_id(self, area_name):
        return self.HH_AREAS.get(area_name)
    
    def set_timeout(self, timeout):
        self.timeout = timeout