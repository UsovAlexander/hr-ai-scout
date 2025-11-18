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
from tqdm import tqdm

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
        self.consecutive_404_errors = 0
        self.max_404_errors = max_404_errors
        
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

    def search_resumes(self, keywords=None, area=1, page=0, items_on_page=20, experience=None):
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
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                self._handle_successful_request()
                return response.text
            elif response.status_code == 404:
                self._handle_404_error()
                return None
            else:
                self._handle_successful_request()
                return None
                
        except requests.exceptions.Timeout:
            self._handle_successful_request()
            return None
        except requests.exceptions.RequestException as e:
            self._handle_successful_request()
            return None
        except StopIteration as e:
            raise e
        except Exception as e:
            self._handle_successful_request()
            return None
    
    def parse_search_results(self, html):
        if not html:
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        resumes = []
        
        resume_elements = soup.find_all('div', {'data-qa': 'resume-serp__resume'})
        
        if not resume_elements:
            resume_elements = soup.find_all('div', {'data-resume-id': True})
        
        for element in resume_elements:
            resume_data = self._parse_resume_card(element)
            if resume_data:
                resumes.append(resume_data)
                
        return resumes
    
    def _parse_experience_to_months(self, experience_text):
        try:
            months = 0
            years_match = re.search(r'(\d+)\s*год', experience_text)
            if years_match:
                months += int(years_match.group(1)) * 12

            years_match = re.search(r'(\d+)\s*лет', experience_text)
            if years_match:
                months += int(years_match.group(1)) * 12
            
            months_match = re.search(r'(\d+)\s*месяц', experience_text)
            if months_match:
                months += int(months_match.group(1))
            
            return months if months > 0 else None
        except Exception:
            return None

    def parse_resume_details(self, url, max_retries=2):
        if self._check_404_limit():
            return {}
            
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    self._handle_successful_request()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    details = {}
                    
                    specialization_block = soup.find_all('li', class_=re.compile(r'resume-block__specialization'))
                    if specialization_block:
                        details['specialization'] = [spec.get_text(strip=True) for spec in specialization_block]
                    
                    experience_block = soup.find('div', {'data-qa': 'resume-block-experience'})
                    if experience_block:
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

                    salary_block = soup.find('span', {'data-qa': 'resume-block-salary'})
                    if salary_block:
                        details['salary'] = salary_block.get_text()
                    
                    skills_block = soup.find('div', {'data-qa': 'skills-table'})
                    if skills_block:
                        skills = skills_block.find_all('div', class_=re.compile(r'bloko-tag bloko-tag_inline'))
                        details['skills'] = [skill.get_text() for skill in skills]
                    
                    education_block = soup.find('div', {'data-qa': 'resume-block-education'})
                    if education_block:
                        education_list = education_block.find_all('div', {'data-qa': 'resume-block-education-name'})
                        if education_list:
                            details['education'] = [edu.get_text() for edu in education_list]
                    
                    courses_block = soup.find('div', {'data-qa': 'resume-block-additional-education'})
                    if courses_block:
                        courses = courses_block.find_all('div', {'data-qa': 'resume-block-education-organization'})
                        details['courses'] = [cor.get_text() for cor in courses]

                    gender = soup.find('span', {'data-qa': 'resume-personal-gender'})
                    if gender:
                        details['gender'] = gender.get_text()

                    location = soup.find('span', {'data-qa': 'resume-personal-address'})
                    if location:
                        details['location'] = location.get_text()
                    
                    return details
                elif response.status_code == 404:
                    self._handle_404_error()
                    return {}
                else:
                    self._handle_successful_request()
                    
            except requests.exceptions.Timeout:
                self._handle_successful_request()
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            except StopIteration as e:
                raise e
            except Exception as e:
                self._handle_successful_request()
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return {}
    
    def _parse_resume_card(self, element):
        try:
            resume = {}
            
            resume_id = element.get('data-resume-id')
            if resume_id:
                resume['id'] = resume_id
            
            link_elem = element.find('a', {'data-qa': 'serp-item__title'})
            if link_elem and link_elem.has_attr('href'):
                href = link_elem.get('href')
                resume['url'] = urljoin(self.base_url, href)
                
                details = self.parse_resume_details(resume['url'])
                resume.update(details)
                
            else:
                if resume_id:
                    resume['url'] = f"{self.base_url}/resume/{resume_id}"
                else:
                    resume['url'] = ''
            
            title_elem = element.find('a', {'data-qa': 'serp-item__title'})
            if not title_elem:
                title_elem = element.find('span', {'data-qa': 'serp-item__title'})
            
            if title_elem and hasattr(title_elem, 'get_text'):
                resume['title'] = title_elem.get_text(strip=True)
            
            age_elem = element.find('span', {'data-qa': 'resume-serp__resume-age'})
            if age_elem and hasattr(age_elem, 'get_text'):
                age_text = age_elem.get_text(strip=True)
                age_numbers = re.findall(r'\d+', age_text)
                if age_numbers:
                    resume['age'] = int(age_numbers[0])
    
            experience_elem = element.find('div', {'data-qa': 'resume-serp_resume-item-total-experience-content'})
            if not experience_elem:
                experience_elem = element.find('span', {'data-qa': 'resume-serp__resume-experience'})
            
            if experience_elem and hasattr(experience_elem, 'get_text'):
                exp_text = experience_elem.get_text()
                resume['total_experience'] = exp_text
                exp_months = self._parse_experience_to_months(exp_text)
                if exp_months:
                    resume['experience_months'] = exp_months
    
            status_elem = element.find('div', class_=re.compile(r'magritte-tag__label'))
            if status_elem and hasattr(status_elem, 'get_text'):
                resume['applicant_status'] = status_elem.get_text(strip=True)
    
            time.sleep(1)
            
            return resume
            
        except Exception:
            return None
    
    def create_dataframe(self, resumes):
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
                'search_query': resume.get('search_query', ''),
                'parsed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            simplified_resumes.append(simple_resume)
        
        df = pd.DataFrame(simplified_resumes)
        return df

    def save_to_files(self, df, base_filename='resumes', resume_name='', include_timestamp=True):
        if df.empty or df is None:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename_suffix = f'_{timestamp}' if timestamp else ''
        
        try:
            csv_filename = f'{base_filename}_{resume_name}{filename_suffix}.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
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
                alter table hh_resumes
                delete where id in {ids}
                """)
            
            clickhouse.insert_dataframe(f"""
                INSERT INTO hh_resumes
                ({', '.join(df.columns)})
                VALUES
            """, df)
            
            print(f"Успешно сохранено {len(df)} записей в ClickHouse таблицу hh_resumes")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении в ClickHouse: {e}")
            return False

    def load_resumes(self, search_terms, areas, pages=1, items_on_page=20, delay=2, timeout=30, use_progress_bar=True):
        if timeout != self.timeout:
            self.timeout = timeout
            
        area_ids = []
        for area in areas:
            if area in self.HH_AREAS:
                area_ids.append(self.HH_AREAS[area])
            else:
                print(f"Регион '{area}' не найден в словаре")
        
        if not area_ids:
            print("Не найдено ни одного валидного региона")
            return None
            
        all_resumes = []
        total_requests = len(search_terms) * len(area_ids) * pages
        
        if use_progress_bar:
            pbar = tqdm(total=total_requests, desc="Сбор резюме")
        
        try:
            for search_term in search_terms:
                if self._check_404_limit():
                    break
                    
                for page in range(pages):
                    if self._check_404_limit():
                        break
                        
                    if use_progress_bar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'Резюме': len(all_resumes),
                            'Запрос': search_term[:20],
                            'Страница': f"{page+1}/{pages}"
                        })
                    
                    html = self.search_resumes(
                        keywords=search_term,
                        area=area_ids,
                        page=page,
                        items_on_page=items_on_page
                    )
                    
                    if html:
                        resumes = self.parse_search_results(html)
                        
                        # Если на текущей странице нет резюме, прекращаем поиск для этого запроса
                        if not resumes:
                            if use_progress_bar:
                                pbar.set_postfix({
                                    'Резюме': len(all_resumes),
                                    'Запрос': search_term[:20],
                                    'Статус': 'Нет резюме - остановка'
                                })
                            break
                        
                        for resume in resumes:
                            resume['search_query'] = search_term
                        
                        all_resumes.extend(resumes)
                        
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        # Если не удалось получить HTML, также прекращаем поиск для этого запроса
                        if use_progress_bar:
                            pbar.set_postfix({
                                'Резюме': len(all_resumes),
                                'Запрос': search_term[:20],
                                'Статус': 'Ошибка запроса - остановка'
                            })
                        break
        
        except StopIteration:
            print("Парсинг прерван из-за превышения лимита ошибок 404")
        
        if use_progress_bar:
            pbar.close()
        
        if all_resumes:
            df = self.create_dataframe(all_resumes)
            print(f"=== ИТОГИ ===")
            print(f"Всего собрано резюме: {len(df)}")
            
            if 'search_query' in df.columns:
                query_stats = df['search_query'].value_counts()
                print("Распределение по поисковым запросам:")
                for query, count in query_stats.items():
                    print(f"  {query}: {count}")
            
            return df
        else:
            print("Не найдено ни одного резюме")
            return None

    def get_available_areas(self):
        return list(self.HH_AREAS.keys())

    def find_area_id(self, area_name):
        return self.HH_AREAS.get(area_name)