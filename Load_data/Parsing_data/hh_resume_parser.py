import requests
import time
import json
import re
import pandas as pd
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlencode, quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hh_resume_parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HHResumeParser')

class HHResumeParser:
    """
    Парсер резюме с сайта hh.ru
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
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logger
        self.consecutive_404_errors = 0  # Счетчик последовательных ошибок 404
        self.max_404_errors = max_404_errors  # Максимальное количество ошибок 404 перед остановкой
        
        # Настройка повторных попыток
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://hh.ru/search/resume',
        })
        self.base_url = 'https://hh.ru'
        
        self.logger.info(f"Парсер инициализирован: timeout={timeout}s, max_retries={max_retries}, max_404_errors={max_404_errors}")
    
    def _check_404_limit(self):
        """Проверка превышения лимита ошибок 404"""
        if self.consecutive_404_errors >= self.max_404_errors:
            self.logger.error(f"Достигнут лимит ошибок 404 ({self.max_404_errors}). Парсинг прекращен.")
            return True
        return False
    
    def _handle_404_error(self):
        """Обработка ошибки 404 и проверка лимита"""
        self.consecutive_404_errors += 1
        self.logger.warning(f"Получена ошибка 404. Всего последовательных ошибок: {self.consecutive_404_errors}/{self.max_404_errors}")
        
        if self._check_404_limit():
            raise StopIteration("Превышен лимит ошибок 404")
    
    def _handle_successful_request(self):
        """Сброс счетчика ошибок при успешном запросе"""
        if self.consecutive_404_errors > 0:
            self.logger.info(f"Сброс счетчика ошибок 404. Было: {self.consecutive_404_errors}")
            self.consecutive_404_errors = 0

    def search_resumes(self, keywords=None, area=1, page=0, items_on_page=20, experience=None):
        """Поиск резюме по параметрам с правильными query-параметрами"""
        # Проверка лимита ошибок перед выполнением запроса
        if self._check_404_limit():
            return None
            
        params = {
            'text': keywords,
            'area': area,
            'pos': 'full_text',
            'logic': 'normal',
            'exp_period': 'all_time',
            'ored_clusters': 'true',
            'order_by': 'relevance',
            'search_period': '0',
            'items_on_page': items_on_page,
            'page': page,
            'hhtmFrom': 'resume_search_result',
            'hhtmFromLabel': 'resume_search_line',
            'customDomain': '1',
            'overRideDomainAreaId': '1',
            'job_search_status': ['active_search', 'looking_for_offers', 'unknown']
        }
        
        # Добавляем опыт работы если указан
        if experience:
            experience_map = {
                'noExperience': 'noExperience',
                'between1And3': 'between1And3',
                'between3And6': 'between3And6',
                'moreThan6': 'moreThan6'
            }
            if experience in experience_map:
                params['experience'] = experience_map[experience]
        
        url = f"{self.base_url}/search/resume"
        self.logger.info(f"Запрос поиска резюме: {url}")
        self.logger.debug(f"Параметры поиска: {params}")
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                self._handle_successful_request()
                self.logger.info("Страница поиска резюме успешно загружена")
                # Проверим наличие результатов
                if "resume-serp__resume" in response.text:
                    self.logger.debug("Найдены резюме в HTML")
                else:
                    self.logger.warning("Резюме не найдены в HTML")
                    # Сохраним для отладки
                    with open('debug_no_results.html', 'w', encoding='utf-8') as f:
                        f.write(response.text)
                return response.text
            elif response.status_code == 404:
                self._handle_404_error()
                self.logger.error("Ошибка 404 - страница не найдена")
                return None
            else:
                self._handle_successful_request()  # Сбрасываем счетчик для других ошибок
                self.logger.error(f"Ошибка при запросе: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            self._handle_successful_request()  # Сбрасываем счетчик для таймаутов
            self.logger.error(f"Таймаут при запросе страницы {page}")
            return None
        except requests.exceptions.RequestException as e:
            self._handle_successful_request()  # Сбрасываем счетчик для других ошибок сети
            self.logger.error(f"Ошибка сети при запросе: {e}")
            return None
        except StopIteration as e:
            # Пробрасываем исключение остановки дальше
            raise e
        except Exception as e:
            self._handle_successful_request()  # Сбрасываем счетчик для других ошибок
            self.logger.error(f"Неожиданная ошибка: {e}")
            return None
    
    def parse_search_results(self, html):
        """Парсинг результатов поиска"""
        if not html:
            self.logger.warning("Пустой HTML для парсинга")
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        resumes = []
        
        # Основной селектор для резюме
        resume_elements = soup.find_all('div', {'data-qa': 'resume-serp__resume'})
        self.logger.info(f"Найдено элементов резюме по data-qa: {len(resume_elements)}")
        
        # Альтернативный поиск по data-resume-id
        if not resume_elements:
            resume_elements = soup.find_all('div', {'data-resume-id': True})
            self.logger.info(f"Найдено элементов по data-resume-id: {len(resume_elements)}")
        
        for i, element in enumerate(resume_elements):
            self.logger.debug(f"Обработка элемента {i+1}/{len(resume_elements)}")
            resume_data = self._parse_resume_card(element)
            if resume_data:
                resumes.append(resume_data)
                self.logger.debug(f"Успешно распарсено резюме: {resume_data.get('title', 'No title')}")
            else:
                self.logger.warning(f"Не удалось распарсить элемент {i+1}")
                
        self.logger.info(f"Всего обработано резюме: {len(resumes)}")
        return resumes
    
    def _parse_experience_to_months(self, experience_text):
        """Конвертация опыта работы в месяцы"""
        try:
            months = 0
            # Опыт в годах
            years_match = re.search(r'(\d+)\s*год', experience_text)
            if years_match:
                months += int(years_match.group(1)) * 12

            years_match = re.search(r'(\d+)\s*лет', experience_text)
            if years_match:
                months += int(years_match.group(1)) * 12
            
            # Опыт в месяцах
            months_match = re.search(r'(\d+)\s*месяц', experience_text)
            if months_match:
                months += int(months_match.group(1))
            
            return months if months > 0 else None
        except Exception as e:
            self.logger.warning(f"Ошибка конвертации опыта '{experience_text}': {e}")
            return None

    def parse_resume_details(self, url, max_retries=2):
        """Парсинг детальной информации со страницы резюме"""
        # Проверка лимита ошибок перед выполнением запроса
        if self._check_404_limit():
            return {}
            
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Загрузка детальной страницы (попытка {attempt + 1}): {url}")
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    self._handle_successful_request()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    details = {}
                    
                    # 1. СПЕЦИАЛИЗАЦИЯ
                    specialization_block = soup.find_all('li', class_=re.compile(r'resume-block__specialization'))
                    if specialization_block:
                        details['specialization'] = [spec.get_text(strip=True) for spec in specialization_block]
                    
                    # 2. ПОСЛЕДНЯЯ КОМПАНИЯ И ДОЛЖНОСТЬ (из опыта работы)
                    experience_block = soup.find('div', {'data-qa': 'resume-block-experience'})
                    if experience_block:
                        # Ищем последнюю компанию
                        company_elems = experience_block.find_all('div', class_=re.compile(r'resume-block-item-gap'))
                        if company_elems:
                            company_elem = company_elems[0].find('div', class_=re.compile(r'bloko-text bloko-text_strong'))
                            if company_elem:
                                details['last_company'] = company_elem.get_text(strip=True)
                            
                            position_elem = company_elems[0].find('div', {'data-qa': 'resume-block-experience-position'})
                            if position_elem:
                                details['last_position'] = position_elem.get_text(strip=True)
                            
                            desc_elem = company_elems[0].find('div', {'data-qa': 'resume-block-experience-description'})
                            if desc_elem:
                                details['last_experience_description'] = desc_elem.get_text(strip=True)
                            
                            period_elem = company_elems[0].find('div', class_=re.compile(r'bloko-text bloko-text_tertiary'))
                            if period_elem:
                                details['last_company_experience_period'] = period_elem.get_text()

                    # 3. ЗАРПЛАТА
                    salary_block = soup.find('span', {'data-qa': 'resume-block-salary'})
                    if salary_block:
                        details['salary'] = salary_block.get_text()
                    
                    # 4. НАВЫКИ
                    skills_block = soup.find('div', {'data-qa': 'skills-table'})
                    if skills_block:
                        skills = skills_block.find_all('div', class_=re.compile(r'bloko-tag bloko-tag_inline'))
                        details['skills'] = [skill.get_text() for skill in skills]
                    
                    # 5. ОБРАЗОВАНИЕ
                    education_block = soup.find('div', {'data-qa': 'resume-block-education'})
                    if education_block:
                        # Основное образование
                        education_list = education_block.find_all('div', {'data-qa': 'resume-block-education-name'})
                        if education_list:
                            details['education'] = [edu.get_text() for edu in education_list]
                    
                    # 6. ПОВЫШЕНИЕ КВАЛИФИКАЦИИ, КУРСЫ
                    courses_block = soup.find('div', {'data-qa': 'resume-block-additional-education'})
                    if courses_block:
                        courses = courses_block.find_all('div', {'data-qa': 'resume-block-education-organization'})
                        details['courses'] = [cor.get_text() for cor in courses]

                    # 7. ПОЛ
                    gender = soup.find('span', {'data-qa': 'resume-personal-gender'})
                    if gender:
                        details['gender'] = gender.get_text()

                    # 8. ЛОКАЦИЯ
                    location = soup.find('span', {'data-qa': 'resume-personal-address'})
                    if location:
                        details['location'] = location.get_text()
                    
                    self.logger.info(f"Успешно собрано деталей: {len(details)}")
                    return details
                elif response.status_code == 404:
                    self._handle_404_error()
                    self.logger.error(f"Ошибка 404 при загрузке детальной страницы")
                    return {}
                else:
                    self._handle_successful_request()
                    self.logger.error(f"Ошибка загрузки страницы: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                self._handle_successful_request()
                self.logger.error(f"Таймаут при загрузке детальной страницы (попытка {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Пауза перед повторной попыткой
                    continue
            except StopIteration as e:
                raise e
            except Exception as e:
                self._handle_successful_request()
                self.logger.error(f"Ошибка при парсинге детальной страницы (попытка {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        self.logger.error("Не удалось загрузить детальную страницу после всех попыток")
        return {}
    
    def _parse_resume_card(self, element):
        """Парсинг карточки резюме в результатах поиска"""
        try:
            resume = {}
            
            # ID резюме из атрибута
            resume_id = element.get('data-resume-id')
            if resume_id:
                resume['id'] = resume_id
            
            # ССЫЛКА НА СТРАНИЦУ РЕЗЮМЕ
            link_elem = element.find('a', {'data-qa': 'serp-item__title'})
            if link_elem and link_elem.has_attr('href'):
                href = link_elem.get('href')
                resume['url'] = urljoin(self.base_url, href)
                
                # ПАРСИМ ДЕТАЛЬНУЮ ИНФОРМАЦИЮ СО СТРАНИЦЫ РЕЗЮМЕ
                details = self.parse_resume_details(resume['url'])
                resume.update(details)
                
            else:
                if resume_id:
                    resume['url'] = f"{self.base_url}/resume/{resume_id}"
                else:
                    resume['url'] = ''
            
            # Заголовок резюме
            title_elem = element.find('a', {'data-qa': 'serp-item__title'})
            if not title_elem:
                title_elem = element.find('span', {'data-qa': 'serp-item__title'})
            
            if title_elem and hasattr(title_elem, 'get_text'):
                resume['title'] = title_elem.get_text(strip=True)
            
            # Возраст
            age_elem = element.find('span', {'data-qa': 'resume-serp__resume-age'})
            if age_elem and hasattr(age_elem, 'get_text'):
                age_text = age_elem.get_text(strip=True)
                age_numbers = re.findall(r'\d+', age_text)
                if age_numbers:
                    resume['age'] = int(age_numbers[0])
    
            # ОБЩИЙ ОПЫТ РАБОТЫ
            experience_elem = element.find('div', {'data-qa': 'resume-serp_resume-item-total-experience-content'})
            if not experience_elem:
                experience_elem = element.find('span', {'data-qa': 'resume-serp__resume-experience'})
            
            if experience_elem and hasattr(experience_elem, 'get_text'):
                exp_text = experience_elem.get_text()
                resume['total_experience'] = exp_text
                exp_months = self._parse_experience_to_months(exp_text)
                if exp_months:
                    resume['experience_months'] = exp_months
    
            # СТАТУС СОИСКАТЕЛЯ
            status_elem = element.find('div', class_=re.compile(r'magritte-tag__label'))
            if status_elem and hasattr(status_elem, 'get_text'):
                resume['applicant_status'] = status_elem.get_text(strip=True)
    
            # Пауза между запросами чтобы не заблокировали
            time.sleep(1)
            
            return resume
            
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге карточки резюме: {e}")
            return None
    
    def create_dataframe(self, resumes):
        """Создание DataFrame из списка резюме"""
        
        simplified_resumes = []
        
        for resume in resumes:
            simple_resume = {
                'id': resume.get('id', ''),
                'title': resume.get('title', ''),
                'url': resume.get('url', ''),
                'specialization': resume.get('specialization', ''),
                'last_company': resume.get('last_company', ''),
                'last_position': resume.get('last_position', ''),
                'last_experience_description': resume.get('last_experience_description', ''),
                'last_company_experience_period': resume.get('last_company_experience_period', ''),
                'skills': resume.get('skills', ''),
                'education': resume.get('education', ''),
                'courses': resume.get('courses', ''),
                'salary': resume.get('salary', ''),
                'age': resume.get('age'),
                'total_experience': resume.get('total_experience', ''),
                'experience_months': resume.get('experience_months'),
                'location': resume.get('location', ''),
                'gender': resume.get('gender', ''),
                'applicant_status': resume.get('applicant_status', ''),
                'parsed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            simplified_resumes.append(simple_resume)
        
        df = pd.DataFrame(simplified_resumes)
        self.logger.info(f"Создан DataFrame с {len(df)} записями")
        return df

    def save_to_files(self, df, base_filename='resumes', resume_name='', include_timestamp=True):
        """Сохранение DataFrame в разные форматы"""
        if df.empty or df is None:
            self.logger.warning("Нет данных для сохранения")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename_suffix = f'_{timestamp}' if timestamp else ''
        
        try:
            # CSV
            csv_filename = f'{base_filename}_{resume_name}{filename_suffix}.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"Данные сохранены в {csv_filename}")
            
            # # Excel
            # excel_filename = f'{base_filename}{filename_suffix}.xlsx'
            # df.to_excel(excel_filename, index=False, engine='openpyxl')
            # self.logger.info(f"Данные сохранены в {excel_filename}")
            
            return csv_filename
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении файлов: {e}")
            return None, None

    def load_resumes(self, search_terms, areas, pages=1, items_on_page=20, delay=2, timeout=30):
        """
        Основная функция для загрузки резюме
        
        Args:
            search_terms (list): Список поисковых запросов
            areas (list): Список регионов (строковые названия)
            pages (int): Количество страниц для парсинга
            delay (int): Задержка между запросами в секундах
            timeout (int): Таймаут для запросов
        
        Returns:
            pd.DataFrame: DataFrame с собранными резюме
        """
        # Обновляем таймаут если нужно
        if timeout != self.timeout:
            self.timeout = timeout
            
        # Конвертируем названия регионов в ID
        area_ids = []
        for area in areas:
            if area in self.HH_AREAS:
                area_ids.append(self.HH_AREAS[area])
            else:
                self.logger.warning(f"Регион '{area}' не найден в словаре")
        
        if not area_ids:
            self.logger.error("Не найдено ни одного валидного региона")
            return None
            
        all_resumes = []
        
        try:
            for search_term in search_terms:
                # Проверка лимита ошибок перед началом нового запроса
                if self._check_404_limit():
                    break
                    
                self.logger.info(f"Поиск по запросу: '{search_term}'")
                for page in range(pages):
                    # Проверка лимита ошибок перед каждой страницей
                    if self._check_404_limit():
                        self.logger.warning("Прерывание парсинга из-за превышения лимита ошибок 404")
                        break
                        
                    self.logger.info(f"Страница {page + 1}/{pages}")
                    
                    html = self.search_resumes(
                        keywords=search_term,
                        area=area_ids,
                        page=page,
                        items_on_page=items_on_page
                    )
                    
                    if html:
                        resumes = self.parse_search_results(html)
                        self.logger.info(f"Найдено резюме для '{search_term}': {len(resumes)}")
                        
                        # Добавляем поисковый запрос к каждому резюме
                        for resume in resumes:
                            resume['search_query'] = search_term
                        
                        all_resumes.extend(resumes)
                        
                        # Пауза между запросами
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        self.logger.warning(f"Не удалось загрузить страницу {page}, пропускаем...")
                        
        except StopIteration as e:
            self.logger.warning(f"Парсинг прерван: {e}")
        
        if all_resumes:
            # Создаем DataFrame
            df = self.create_dataframe(all_resumes)
            
            self.logger.info(f"=== ИТОГИ ===")
            self.logger.info(f"Всего собрано резюме: {len(df)}")
            self.logger.info(f"Колонки: {list(df.columns)}")
            
            # Статистика по запросам
            if 'search_query' in df.columns:
                query_stats = df['search_query'].value_counts()
                self.logger.info("Распределение по поисковым запросам:")
                for query, count in query_stats.items():
                    self.logger.info(f"  {query}: {count}")
            
            return df
        else:
            self.logger.warning("Не найдено ни одного резюме")
            return None

    def get_available_areas(self):
        """Возвращает список доступных регионов"""
        areas = list(self.HH_AREAS.keys())
        self.logger.info(f"Доступно регионов: {len(areas)}")
        return areas

    def find_area_id(self, area_name):
        """Находит ID региона по названию"""
        area_id = self.HH_AREAS.get(area_name)
        if area_id is not None:
            self.logger.debug(f"Найден ID региона '{area_name}': {area_id}")
        else:
            self.logger.warning(f"Регион '{area_name}' не найден")
        return area_id