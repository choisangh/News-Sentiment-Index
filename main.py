import pandas as pd
from src.doubleHP import double_HP
from src.x13ARIMA import x13ARIMA
from model import labeling

DATA_ROUTE = './data/data.csv'


newsdf = pd.read_csv(DATA_ROUTE)

newsdf['분류'] = newsdf['본문'].apply(labeling)
newsdf['일자'] = pd.to_datetime(newsdf['일자'], format='%Y%m%d')
newsdf = newsdf.set_index('일자')

negative = newsdf[newsdf['분류'] == 'negative']
positive = newsdf[newsdf['분류'] == 'positive']
negative = negative.resample('1D').count()  # 일 단위 빈도수 리샘플링
negative = negative.resample('1M').sum()  # 월 단위 리샘플링
positive = positive.resample('1D').count()  # 일 단위 빈도수 리샘플링
positive = positive.resample('1M').sum()  # 월 단위 리샘플링


news_index = pd.merge(negative[['분류']], positive[['분류']], left_index=True, right_index=True, how='outer')
final = news_index.fillna(0)
news_index['value'] = (((news_index['positive']-news_index['negative'])/(news_index['positive']+news_index['negative']))
                       * 100 + 100)

news_index['value'] = double_HP(news_index['value'], 0.5)
