import requests
import time
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from clickhouse_driver import Client as ClickhouseClient

class HHVacancyParser:
    """
    Парсер вакансий с сайта hh.ru через API
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
        self.base_url = 'https://api.hh.ru/vacancies'
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
            'Accept': 'application/json',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
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

    def search_vacancies(self, search_text='Аналитик данных', area=1, page=0, per_page=100):
        if self._check_404_limit():
            return None
            
        params = {
            'text': search_text,
            'area': area,
            'page': page,
            'per_page': per_page
        }
        
        try:
            response = self.session.get(
                self.base_url, 
                params=params, 
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                self._handle_successful_request()
                return response.json()
            elif response.status_code == 400:
                self._handle_successful_request()
                return None
            elif response.status_code == 403:
                self._handle_successful_request()
                return None
            elif response.status_code == 404:
                self._handle_404_error()
                return None
            else:
                self._handle_successful_request()
                return None
                
        except requests.exceptions.Timeout:
            self._handle_successful_request()
            return None
        except requests.exceptions.ConnectionError:
            self._handle_successful_request()
            return None
        except requests.exceptions.RequestException:
            self._handle_successful_request()
            return None
        except StopIteration as e:
            raise e
        except json.JSONDecodeError:
            self._handle_successful_request()
            return None
        except Exception:
            self._handle_successful_request()
            return None
    
    def parse_vacancies_page(self, json_data):
        vacancies = []
        
        if not json_data:
            return vacancies
        
        if 'items' not in json_data:
            return vacancies
        
        items = json_data['items']
        if not items:
            return vacancies
        
        for item in items:
            try:
                vacancy = {
                    'id': item.get('id', ''),
                    'name': item.get('name', ''),
                    'area': item.get('area', {}).get('name', ''),
                    'url': item.get('url', ''),
                    'alternate_url': item.get('alternate_url', ''),
                    'requirement': item.get('snippet', {}).get('requirement', ''),
                    'responsibility': item.get('snippet', {}).get('responsibility', ''),
                    'employer': item.get('employer', {}).get('name', ''),
                    'experience': item.get('experience', {}).get('name', ''),
                    'employment': item.get('employment', {}).get('name', ''),
                    'schedule': item.get('schedule', {}).get('name', ''),
                    'published_at': item.get('published_at', ''),
                    'created_at': item.get('created_at', ''),
                }
                
                salary = item.get('salary')
                if salary and isinstance(salary, dict):
                    vacancy.update({
                        'salary_from': salary.get('from'),
                        'salary_to': salary.get('to'),
                        'salary_currency': salary.get('currency'),
                        'salary_gross': salary.get('gross')
                    })
                else:
                    vacancy.update({
                        'salary_from': None,
                        'salary_to': None,
                        'salary_currency': None,
                        'salary_gross': None
                    })

                details = self.get_vacancy_details(vacancy['id'], vacancy['url'])
                vacancy.update(details)
                
                if vacancy['requirement']:
                    vacancy['requirement'] = self._clean_html_tags(vacancy['requirement'])
                if vacancy['responsibility']:
                    vacancy['responsibility'] = self._clean_html_tags(vacancy['responsibility'])
                if vacancy['description']:
                    vacancy['description'] = self._clean_html_tags(vacancy['description'])
                
                vacancies.append(vacancy)
                
            except Exception:
                continue
        
        return vacancies
    
    def _clean_html_tags(self, text):
        if not text:
            return text
        return re.sub(r'<[^>]+>', '', text)
    
    def load_vacancies(self, search_terms, areas, pages=1, per_page=100, delay=1, 
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
                                'Ошибки': error_count,
                                'Страница': f"{page+1}/{pages}"
                            })
                        
                        json_data = self.search_vacancies(
                            search_text=search_term,
                            area=area_id,
                            page=page,
                            per_page=per_page
                        )
                        
                        if json_data:
                            vacancies = self.parse_vacancies_page(json_data)
                            
                            for vacancy in vacancies:
                                vacancy['search_query'] = search_term
                                vacancy['area_id'] = area_id
                            
                            all_vacancies.extend(vacancies)
                            
                            total_pages = json_data.get('pages', 1)
                            found = json_data.get('found', 0)
                            
                            if page >= total_pages - 1:
                                break
                                
                            if stop_on_rate_limit and found == 0 and page == 0:
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
        
        df['parsed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype('int64')
        
        date_columns = ['published_at', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.tz_localize(None)
        
        salary_columns = ['salary_from', 'salary_to']
        for col in salary_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def save_to_files(self, df, base_filename='vacancies', vacancy_name='', include_timestamp=True):
        if df.empty or df is None:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
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
        if df.empty:
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
                    item = response.json()
                    details['description'] = item['description']
                    return details
                elif response.status_code == 404:
                    return None
                else:
                    pass
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return None
    
    def get_available_areas(self):
        return list(self.HH_AREAS.keys())
    
    def find_area_id(self, area_name):
        return self.HH_AREAS.get(area_name)
    
    def set_timeout(self, timeout):
        self.timeout = timeout