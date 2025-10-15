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
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HHVacancyParser')

class HHVacancyParser:
    """
    Парсер вакансий с сайта hh.ru через API
    с обработкой таймаутов и повторными попытками
    """
    
    # Словарь регионов HH
    HH_AREAS = {
        # Россия (основные города)
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
        
        # Другие страны СНГ
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
        
        # Европа
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
        
        # Америка
        "США": 10644,
        "Канада": 10645,
        "Бразилия": 10646,
        "Мексика": 10647,
        "Аргентина": 10648,
        
        # Азия
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
        
        # Другие
        "Австралия": 10662,
        "Новая Зеландия": 10663,
        "ЮАР": 10664,
        
        # Специальные значения
        "Все регионы": 0,
        "Города с населением более 1 млн": 1001,
        "Города с населением 500 тыс. - 1 млн": 1002,
        "Регионы России (кроме Москвы и СПб)": 1017,
        "Другие регионы": 1024,
        "Удаленная работа": 1036,
        "Россия": 113
    }

    def __init__(self, timeout=30, max_retries=3, max_404_errors=5):
        """
        Инициализация парсера
        
        Args:
            timeout (int): Таймаут для запросов в секундах
            max_retries (int): Максимальное количество повторных попыток
            max_404_errors (int): Максимальное количество ошибок 404 перед остановкой
        """
        self.base_url = 'https://api.hh.ru/vacancies'
        self.timeout = timeout
        self.session = requests.Session()
        self.consecutive_404_errors = 0
        self.max_404_errors = max_404_errors
        
        # Настройка стратегии повторных попыток
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        
        # Настройка адаптера с повторными попытками
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Настройка заголовков
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
        })
        
        logger.info(f"Парсер инициализирован: timeout={timeout}s, max_retries={max_retries}, max_404_errors={max_404_errors}")
    
    def _check_404_limit(self):
        """Проверка превышения лимита ошибок 404"""
        if self.consecutive_404_errors >= self.max_404_errors:
            logger.error(f"Достигнут лимит ошибок 404 ({self.max_404_errors}). Парсинг прекращен.")
            return True
        return False
    
    def _handle_404_error(self):
        """Обработка ошибки 404 и проверка лимита"""
        self.consecutive_404_errors += 1
        logger.warning(f"Получена ошибка 404. Всего последовательных ошибок: {self.consecutive_404_errors}/{self.max_404_errors}")
        
        if self._check_404_limit():
            raise StopIteration("Превышен лимит ошибок 404")
    
    def _handle_successful_request(self):
        """Сброс счетчика ошибок при успешном запросе"""
        if self.consecutive_404_errors > 0:
            logger.info(f"Сброс счетчика ошибок 404. Было: {self.consecutive_404_errors}")
            self.consecutive_404_errors = 0

    def search_vacancies(self, search_text='Аналитик данных', area=1, page=0, per_page=100):
        """
        Поиск вакансий через API HH с обработкой ошибок
        
        Args:
            search_text (str): Текст для поиска
            area (int): ID региона
            page (int): Номер страницы
            per_page (int): Количество вакансий на странице
        
        Returns:
            dict: JSON ответ от API или None в случае ошибки
        """
        # Проверка лимита ошибок перед выполнением запроса
        if self._check_404_limit():
            return None
            
        params = {
            'text': search_text,
            'area': area,
            'page': page,
            'per_page': per_page
        }
        
        logger.debug(f"Поиск вакансий: {params}")
        
        try:
            response = self.session.get(
                self.base_url, 
                params=params, 
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                self._handle_successful_request()
                logger.info(f"Страница {page} успешно загружена ({len(response.json().get('items', []))} вакансий)")
                return response.json()
            elif response.status_code == 400:
                self._handle_successful_request()
                logger.warning(f"Некорректный запрос: {response.json().get('description', 'Unknown error')}")
                return None
            elif response.status_code == 403:
                self._handle_successful_request()
                logger.error("Доступ запрещен. Возможно, превышен лимит запросов.")
                return None
            elif response.status_code == 404:
                self._handle_404_error()
                logger.warning("Страница не найдена")
                return None
            else:
                self._handle_successful_request()
                logger.warning(f"Ошибка API: {response.status_code} - {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            self._handle_successful_request()
            logger.error(f"Таймаут при запросе страницы {page} (timeout={self.timeout}s)")
            return None
        except requests.exceptions.ConnectionError:
            self._handle_successful_request()
            logger.error("Ошибка подключения к API HH")
            return None
        except requests.exceptions.RequestException as e:
            self._handle_successful_request()
            logger.error(f"Ошибка сети: {e}")
            return None
        except StopIteration as e:
            raise e
        except json.JSONDecodeError as e:
            self._handle_successful_request()
            logger.error(f"Ошибка декодирования JSON: {e}")
            return None
        except Exception as e:
            self._handle_successful_request()
            logger.error(f"Неожиданная ошибка: {e}")
            return None
    
    def parse_vacancies_page(self, json_data):
        """
        Парсинг страницы с вакансиями с обработкой ошибок
        
        Args:
            json_data (dict): JSON данные от API
        
        Returns:
            list: Список словарей с данными вакансий
        """
        vacancies = []
        
        if not json_data:
            logger.warning("Нет данных для парсинга")
            return vacancies
        
        if 'items' not in json_data:
            logger.warning("В ответе API отсутствует ключ 'items'")
            return vacancies
        
        items = json_data['items']
        if not items:
            logger.info("На странице нет вакансий")
            return vacancies
        
        for i, item in enumerate(items):
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
                
                # Обработка зарплаты с проверкой на None
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
                
                # Очистка HTML тегов
                if vacancy['requirement']:
                    vacancy['requirement'] = self._clean_html_tags(vacancy['requirement'])
                if vacancy['responsibility']:
                    vacancy['responsibility'] = self._clean_html_tags(vacancy['responsibility'])
                if vacancy['description']:
                    vacancy['description'] = self._clean_html_tags(vacancy['description'])
                
                vacancies.append(vacancy)
                
            except Exception as e:
                logger.warning(f"Ошибка при парсинге вакансии {i}: {e}")
                continue
        
        logger.info(f"Успешно обработано вакансий: {len(vacancies)}/{len(items)}")
        return vacancies
    
    def _clean_html_tags(self, text):
        """Очистка HTML тегов из текста"""
        if not text:
            return text
        return re.sub(r'<[^>]+>', '', text)
    
    def load_vacancies(self, search_terms, areas, pages=1, per_page=100, delay=1, 
                      use_progress_bar=True, stop_on_rate_limit=True):
        """
        Основная функция для загрузки вакансий с улучшенной обработкой ошибок
        
        Args:
            search_terms (list): Список поисковых запросов
            areas (list): Список регионов (строковые названия)
            pages (int): Количество страниц для парсинга
            per_page (int): Количество вакансий на странице
            delay (int): Задержка между запросами в секундах
            use_progress_bar (bool): Показывать прогресс-бар
            stop_on_rate_limit (bool): Останавливаться при достижении лимита запросов
        
        Returns:
            pd.DataFrame: DataFrame с собранными вакансиями
        """
        # Конвертируем названия регионов в ID
        area_ids = []
        invalid_areas = []
        
        for area in areas:
            if area in self.HH_AREAS:
                area_ids.append(self.HH_AREAS[area])
            else:
                invalid_areas.append(area)
                logger.warning(f"Регион '{area}' не найден в словаре")
        
        if invalid_areas:
            logger.warning(f"Невалидные регионы: {invalid_areas}")
        
        if not area_ids:
            logger.error("Не найдено ни одного валидного региона")
            return None
            
        all_vacancies = []
        total_requests = len(search_terms) * len(area_ids) * pages
        
        logger.info(f"Начало сбора вакансий: {len(search_terms)} запросов, {len(area_ids)} регионов, {pages} страниц")
        
        # Создаем прогресс-бар если нужно
        if use_progress_bar:
            pbar = tqdm(total=total_requests, desc="Сбор вакансий")
        
        request_count = 0
        error_count = 0
        
        try:
            for search_term in search_terms:
                # Проверка лимита ошибок перед новым запросом
                if self._check_404_limit():
                    logger.warning("Прерывание парсинга из-за превышения лимита ошибок 404")
                    break
                    
                logger.info(f"Поиск по запросу: '{search_term}'")
                
                for area_id in area_ids:
                    # Проверка лимита ошибок перед новым регионом
                    if self._check_404_limit():
                        break
                        
                    area_name = self._get_area_name(area_id)
                    logger.info(f"Регион: {area_name} (ID: {area_id})")
                    
                    for page in range(pages):
                        # Проверка лимита ошибок перед каждой страницей
                        if self._check_404_limit():
                            logger.warning("Прерывание парсинга из-за превышения лимита ошибок 404")
                            break
                            
                        request_count += 1
                        
                        if use_progress_bar:
                            pbar.update(1)
                            pbar.set_postfix({
                                'Вакансии': len(all_vacancies),
                                'Ошибки': error_count,
                                'Страница': f"{page+1}/{pages}"
                            })
                        
                        logger.debug(f"Запрос: '{search_term}', регион {area_id}, страница {page+1}")
                        
                        json_data = self.search_vacancies(
                            search_text=search_term,
                            area=area_id,
                            page=page,
                            per_page=per_page
                        )
                        
                        if json_data:
                            vacancies = self.parse_vacancies_page(json_data)
                            logger.info(f"Найдено вакансий для '{search_term}': {len(vacancies)}")
                            
                            # Добавляем поисковый запрос к каждой вакансии
                            for vacancy in vacancies:
                                vacancy['search_query'] = search_term
                                vacancy['area_id'] = area_id
                            
                            all_vacancies.extend(vacancies)
                            
                            # Проверка на последнюю страницу
                            total_pages = json_data.get('pages', 1)
                            found = json_data.get('found', 0)
                            
                            if page >= total_pages - 1:
                                logger.info(f"Достигнута последняя страница. Всего найдено: {found} вакансий")
                                break
                                
                            # Проверка на лимит запросов
                            if stop_on_rate_limit and found == 0 and page == 0:
                                logger.warning("Возможно достигнут лимит запросов. Прерывание.")
                                break
                        
                        else:
                            error_count += 1
                            logger.warning(f"Не удалось загрузить страницу {page+1}")
                        
                        # Пауза между запросами
                        if delay > 0:
                            time.sleep(delay)
        
        except StopIteration as e:
            logger.warning(f"Парсинг прерван: {e}")
        
        if use_progress_bar:
            pbar.close()
        
        logger.info(f"Сбор завершен. Запросов: {request_count}, Ошибок: {error_count}")
        
        if all_vacancies:
            # Создаем DataFrame
            df = self.create_dataframe(all_vacancies)
            
            logger.info(f"=== ИТОГИ ===")
            logger.info(f"Всего собрано вакансий: {len(df)}")
            logger.info(f"Уникальных вакансий: {df['id'].nunique()}")
            
            # Статистика по запросам
            if 'search_query' in df.columns:
                logger.info("Распределение по поисковым запросам:")
                for query, count in df['search_query'].value_counts().items():
                    logger.info(f"  {query}: {count}")
            
            return df
        else:
            logger.warning("Не найдено ни одной вакансии")
            return None
    
    def _get_area_name(self, area_id):
        """Получение названия региона по ID"""
        for name, id_ in self.HH_AREAS.items():
            if id_ == area_id:
                return name
        return f"Unknown ({area_id})"
    
    def create_dataframe(self, vacancies):
        """
        Создание DataFrame из списка вакансий
        
        Args:
            vacancies (list): Список словарей с данными вакансий
        
        Returns:
            pd.DataFrame: DataFrame с вакансиями
        """
        df = pd.DataFrame(vacancies)
        
        if df.empty:
            return df
        
        # Добавляем timestamp
        df['parsed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Конвертируем ID в int если возможно
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype('int64')
        
        # Конвертируем даты - убираем временные зоны для совместимости с Excel
        date_columns = ['published_at', 'created_at']
        for col in date_columns:
            if col in df.columns:
                # Конвертируем в datetime и убираем временную зону
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Преобразуем в наивные datetime (без временной зоны)
                df[col] = df[col].dt.tz_localize(None)
        
        # Конвертируем зарплаты
        salary_columns = ['salary_from', 'salary_to']
        for col in salary_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Создан DataFrame с {len(df)} вакансиями и {len(df.columns)} колонками")
        return df
    
    def save_to_files(self, df, base_filename='vacancies', vacancy_name='', include_timestamp=True):
        """
        Сохранение DataFrame в разные форматы
        
        Args:
            df (pd.DataFrame): DataFrame для сохранения
            base_filename (str): Базовое имя файла
            include_timestamp (bool): Добавлять timestamp к имени файла
        
        Returns:
            tuple: Пути к сохраненным файлам
        """
        if df.empty or df is None:
            logger.warning("Нет данных для сохранения")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename_suffix = f'_{timestamp}' if timestamp else ''
        
        
        try:
            # Создаем копию DataFrame для безопасного сохранения
            df_save = df.copy()
            
            # Убеждаемся, что все datetime колонки без временных зон
            datetime_columns = df_save.select_dtypes(include=['datetime64[ns]']).columns
            for col in datetime_columns:
                df_save[col] = df_save[col].dt.tz_localize(None)
            
            # CSV
            csv_filename = f'{base_filename}_{vacancy_name}{filename_suffix}.csv'
            df_save.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            logger.info(f"Данные сохранены в CSV: {csv_filename}")
            
            # # Excel
            # excel_filename = f'{base_filename}{filename_suffix}.xlsx'
            
            # # Создаем Excel writer с настройками
            # with pd.ExcelWriter(excel_filename, engine='openpyxl', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
            #     df_save.to_excel(writer, index=False, sheet_name='Vacancies')
                
            #     # Получаем workbook и worksheet для дополнительных настроек
            #     workbook = writer.book
            #     worksheet = writer.sheets['Vacancies']
                
            #     # Настраиваем ширину колонок для лучшего отображения
            #     for column in worksheet.columns:
            #         max_length = 0
            #         column_letter = column[0].column_letter
            #         for cell in column:
            #             try:
            #                 if len(str(cell.value)) > max_length:
            #                     max_length = len(str(cell.value))
            #             except:
            #                 pass
            #         adjusted_width = min(max_length + 2, 50)
            #         worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # logger.info(f"Данные сохранены в Excel: {excel_filename}")
            
            return csv_filename
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении файлов: {e}")
            # Попробуем сохранить только CSV как запасной вариант
            try:
                csv_filename = f'{base_filename}{filename_suffix}_backup.csv'
                df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                logger.info(f"Резервная копия сохранена в CSV: {csv_filename}")
                return csv_filename, None
            except Exception as backup_error:
                logger.error(f"Не удалось сохранить даже резервную копию: {backup_error}")
                return None, None
    
    def get_vacancy_details(self, vacancy_id, url, max_retries=2):
        """
        Получение детальной информации по конкретной вакансии
        
        Args:
            vacancy_id (str): ID вакансии
            max_retries (int): Максимальное количество попыток
        
        Returns:
            dict: Детальная информация о вакансии
        """
        details = {}
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Запрос деталей вакансии {vacancy_id} (попытка {attempt + 1})")
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    logger.info(f"Детали вакансии {vacancy_id} успешно загружены")
                    item = response.json()
                    details['description'] = item['description']
                    return details
                elif response.status_code == 404:
                    logger.warning(f"Вакансия {vacancy_id} не найдена")
                    return None
                else:
                    logger.warning(f"Ошибка при запросе деталей вакансии: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Таймаут при запросе деталей вакансии {vacancy_id}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Экспоненциальная backoff задержка
                    continue
            except Exception as e:
                logger.error(f"Ошибка при получении деталей вакансии: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        logger.error(f"Не удалось загрузить детали вакансии {vacancy_id} после {max_retries} попыток")
        return None
    
    def get_available_areas(self):
        """Возвращает список доступных регионов"""
        return list(self.HH_AREAS.keys())
    
    def find_area_id(self, area_name):
        """Находит ID региона по названию"""
        return self.HH_AREAS.get(area_name)
    
    def set_timeout(self, timeout):
        """Установка нового значения таймаута"""
        self.timeout = timeout
        logger.info(f"Таймаут установлен: {timeout}s")