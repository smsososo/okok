from dotenv import load_dotenv
import os

# 加載 .env 文件中的環境變量
load_dotenv()

# 獲取 OpenAI API 密鑰
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    raise ValueError("OPENAI_API_KEY is not set in the environment or .env file")


import base64
from datetime import datetime, timedelta
import streamlit as st            
import time
import sqlite3
from Data_pulling import generate_random_data_for_over_consumption, update_consumption_data, add_random_consumption, \
    generate_random_data_for_households, add_households_consumption, generate_random_data_for_Solar, \
    add_solar_consumption, update_households_data, update_solar_data
from passlib.hash import pbkdf2_sha256
import re
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm        
import plotly.express as px
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI    
import pandas as pd
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # 從 .env 文件中加載環境變量
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
user_good_name=""
image_data=""
overall_consumption_data=None
house_holds_data=None
solar_data=None

#變量正確加載 
# 這段代碼將所有數據加載操作合併到一個 try 區塊中，# 如果任何一個數據加載失敗，將捕獲異常並打印錯誤信息，同時將相應的變量設置為 None。
# 這樣可以確保您的代碼更簡潔，且更容易調試。
try:
    overall_consumption_data = pd.read_excel(f"Data/{st.session_state.username}-over-consumption.xlsx")
    overall_consumption_data['Date'] = pd.to_datetime(overall_consumption_data['Date'])
    update_consumption_data(f"Data/{st.session_state.username}-over-consumption.xlsx")
    print("Overall consumption data loaded successfully")

    house_holds_data = pd.read_excel(f"Data/{st.session_state.username}-households-data.xlsx")
    house_holds_data['Date'] = pd.to_datetime(house_holds_data['Date'])
    update_households_data(f"Data/{st.session_state.username}-households-data.xlsx")
    print("Households data loaded successfully")

    solar_data = pd.read_excel(f"Data/{st.session_state.username}-solar-data.xlsx")
    solar_data['Date'] = pd.to_datetime(solar_data['Date'])
    update_solar_data(f"Data/{st.session_state.username}-solar-data.xlsx")
    print("Solar data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    overall_consumption_data = None
    house_holds_data = None
    solar_data = None



# 創建一個 SQLite 數據庫連接
conn = sqlite3.connect('Database.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        good_name TEXT,
        contact_gmail TEXT,
        profile TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')


if hasattr(st.session_state, "logged_in") and st.session_state.logged_in and not hasattr(st.session_state, "isGenerated"):
    end_date = datetime.now()
    try:
        overall_consumption_data = pd.read_excel(f"Data/{st.session_state.username}-over-consumption.xlsx")
        house_holds_data = pd.read_excel(f"Data/{st.session_state.username}-households-data.xlsx")
        solar_data = pd.read_excel(f"Data/{st.session_state.username}-solar-data.xlsx")
    except:
        overall_consumption_data = pd.read_excel(f"Data/Consumption_data.xlsx")
        start_date = end_date - timedelta(days=120)  # 過去 3 年
        generated_data = generate_random_data_for_over_consumption(start_date, end_date, overall_consumption_data)
        overall_consumption_data =generated_data #pd.concat([overall_consumption_data, generated_data], ignore_index=True)
        overall_consumption_data['Date'] = pd.to_datetime(overall_consumption_data['Date'])
        overall_consumption_data.to_excel(f'Data/{st.session_state.username}-over-consumption.xlsx',index=False)

        house_holds_data = pd.read_excel(f"Data/households_data.xlsx")
        start_date = end_date - timedelta(days=120)  # 過去 3 年
        generated_data = generate_random_data_for_households(start_date, end_date, house_holds_data)
        house_holds_data = generated_data  # pd.concat([overall_consumption_data, generated_data], ignore_index=True)
        house_holds_data['Date'] = pd.to_datetime(house_holds_data['Date'])
        house_holds_data.to_excel(f"Data/{st.session_state.username}-households-data.xlsx",index=False)

        solar_data = pd.read_excel(f"Data/Solar_Consumption_and_Dataset.xlsx")
        start_date = end_date - timedelta(days=120)  # 過去 3 年
        generated_data = generate_random_data_for_Solar(start_date, end_date, solar_data)
        solar_data = generated_data  # pd.concat([overall_consumption_data, generated_data], ignore_index=True)
        solar_data['Date'] = pd.to_datetime(solar_data['Date'])
        solar_data.to_excel(f"Data/{st.session_state.username}-solar-data.xlsx", index=False)

        st.session_state.isGenerated=True

def run_chain(agent,query):
    return agent.invoke(query)

def delete_user_by_email(email):
    try:
        # 定義刪除查詢
        delete_query = '''
            DELETE FROM users
            WHERE contact_gmail = ?
        '''
        cursor.execute(delete_query, (email,))
        # 提交對數據庫的更改
        conn.commit()

        return f"帳戶刪除成功。"

    except sqlite3.Error as e:
        return "刪除用戶時出錯: {e}"

# 註冊新用戶的函數
def register(image_data,username, password, good_name, contact_gmail):
    hashed_password = pbkdf2_sha256.hash(password)
    cursor.execute('''
        INSERT INTO users (username, password, good_name,contact_gmail, profile)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, hashed_password, good_name, contact_gmail,image_data))
    conn.commit()

# 驗證密碼的函數
def is_valid_password(password):
    # 最小長度為 6 個字符
    if len(password) < 6:
        return False


    # 如果所有條件都滿足，則密碼有效
    return True

# 認證用戶的函數
def authenticate(username, password):
    cursor.execute('SELECT password, good_name FROM users WHERE contact_gmail = ?', (username,))
    result = cursor.fetchone()
    if result:
        hashed_password = result[0]
        return pbkdf2_sha256.verify(password, hashed_password),result[1]
    return False,""

# 檢查用戶名是否存在的函數
def is_username_taken(username):
    cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()[0]
    conn.commit()
    return result > 0

# 檢查 Gmail 地址是否已經註冊的函數
def is_gmail_taken(gmail):
    cursor.execute('SELECT COUNT(*) FROM users WHERE contact_gmail = ?', (gmail,))
    result = cursor.fetchone()[0]
    conn.commit()
    return result > 0

def Login():
    global user_good_name

    st.header("登錄到您的帳戶")
    # 登錄輸入字段
    with st.form("login_form"):
        login_username = st.text_input("您的電子郵件地址")
        login_password = st.text_input("密碼", type="password")
        # 登錄按鈕
        if st.form_submit_button("登錄"):
            if login_username and login_password:
                isauthenticated,user_good_name=authenticate(login_username, login_password)
                if isauthenticated:
                        st.success(f"歡迎 {user_good_name}，您已成功登錄！")
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.user_good_name = user_good_name
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error("登錄失敗。請檢查您的憑證或該憑證可能已經存在帳戶。")

            else:
                st.warning("請填寫所有字段並重試。")


def Register():
    st.header("註冊您的帳戶")
    with st.form("profile_form"):
        # 註冊輸入字段
        profile_image = st.file_uploader("上傳個人資料圖片", type=["jpg", "jpeg", "png"])
        register_username = st.text_input("用戶名")
        good_name = st.text_input("好名字")
        contact_gmail = st.text_input("您的 Gmail")
        validation_message = """
        密碼必須符合以下標準：
        - 最小長度為 6 個字符。
        請確保您的密碼符合這些要求以提高安全性。
        """
        st.write(validation_message)
        register_password = st.text_input("密碼", type="password")
        confirm_password = st.text_input("確認密碼", type="password")

        register_button = st.form_submit_button('註冊帳戶')
        
        # 驗證輸入和註冊按鈕
        if register_button:
            if is_username_taken(register_username):
                st.error("用戶名已被佔用。請選擇其他用戶名。")
                return
            if is_gmail_taken(contact_gmail):
                st.error("Gmail 地址已被註冊。請使用其他 Gmail 地址。")
                return

            if (profile_image and register_username and register_password and confirm_password and good_name and contact_gmail):
                if register_password == confirm_password and is_valid_password(register_password):
                    image_data = None
                    if profile_image:
                        # 將圖片轉換為 base64 並存儲
                        image_data = base64.b64encode(profile_image.read()).decode("utf-8")
                    register(image_data, register_username, register_password, good_name, contact_gmail)
                    st.success("註冊成功！您現在可以登錄了。")
                else:
                    st.error("密碼和確認密碼不匹配或不符合標準。")
            else:
                st.warning("請填寫所有字段並上傳個人資料圖片。")
def home():
    st.image("Images/Lgo-Image-Eletric.png", width=50)
    st.title("解鎖知情能源管理的力量")
    st.write(
        "通過我們的先進儀表板，輕鬆跟踪使用趨勢，識別優化區域，獲得對家庭能源消耗和消耗模式的全面可視性。")

def get_user_profile_by_email(email_address):

    cursor.execute('''
        SELECT username,good_name,profile,timestamp
        FROM users
        WHERE contact_gmail = ?
    ''', (email_address,))
    user_profile = cursor.fetchone()
    if user_profile:
        # 找到用戶資料
        return {
            'username': user_profile[0],
            'good_name': user_profile[1],
            'contact_gmail':email_address,
            'profile':user_profile[2],
            'timestamp':user_profile[3],
        }
    else:
        # 未找到用戶資料
        return None

def profile():
    if hasattr(st.session_state, "logged_in") and st.session_state.logged_in:
        st.title("我的個人資料")
        st.write("輕鬆編輯您的個人資料信息。")
        st.markdown("---")
        # 從會話變量中獲取用戶電子郵件（替換為您的會話邏輯）
        user_email = st.session_state.username
        if not user_email:
            st.error("會話中未找到用戶電子郵件。")
            return
        # 根據電子郵件檢索用戶資料信息
        user_profile = get_user_profile_by_email(user_email)
        if user_profile is None:
            st.error("未找到用戶。")
            return
        # 使用 st.form 進行更有組織的佈局
        with st.form("profile_form"):
            # 允許用戶查看或編輯其個人資料

            image_data = base64.b64decode(user_profile['profile'])
            hover_code = """
                        <style>
                        .image-holder 
                        {
                            text-align:center;
                            margin-bottom:30px;
                        }
                        .image
                        {
                        border-radius:50%;
                        width:20%;
                        cursor:pointer;
                        }
                            .image:hover {

                                transform: scale(1.3);  /* 您可以根據需要調整縮放比例 */
                                background-color:rgba(244,85,85,255);
                                transition:0.5s all;

                            }
                           .msg-hovering:hover
                           {
                           cursor:pointer;
                           transform:scale(0.95);
                           transition:0.3 s all;
                           }
                        </style>
                        """
            image_code = hover_code + f'<div class="image-holder" ><img class="image" src="data:image/png;base64,{base64.b64encode(image_data).decode()}" alt="{user_profile["good_name"]} 個人資料" ></div>'
            st.markdown(image_code, unsafe_allow_html=True)
            st.subheader("信息")
            edited_contact_gmail = st.text_input("聯繫 Gmail", value=user_profile['contact_gmail'], disabled=True)
            edited_timeStamp = st.text_input("帳戶創建時間戳", value=user_profile['timestamp'],
                                             disabled=True)

            edited_user_name = st.text_input("用戶名", value=user_profile['username'])
            edited_good_name = st.text_input("好名字", value=user_profile['good_name'])
            col1, col2 = st.columns([1, 1])
            with col1:
                update_button = st.form_submit_button("更新個人資料信息",
                                                      use_container_width=True)
                if update_button:
                    if edited_user_name and edited_good_name and edited_contact_gmail:
                        update_profile(edited_user_name, edited_good_name, edited_contact_gmail)
                        st.success("個人資料更新成功！")
                    else:
                        st.warning('請填寫所有字段並且不要留空。')
            with col2:
                delete_button = st.form_submit_button("刪除我的帳戶", use_container_width=True)
                if delete_button:
                    results = delete_user_by_email(st.session_state.username)
                    st.error(f'很抱歉看到你離開。 ({results})')
                    st.session_state.logged_in = False
                    st.session_state.username = None
                    time.sleep(1)
                    st.rerun()
    else:
        st.title("請使用適當的帳戶登錄")

def update_profile(edited_user_name,edited_good_name, edited_contact_gmail ):
    cursor.execute('''
            UPDATE users
            SET username=?,good_name = ?
            WHERE contact_gmail = ?
        ''', (edited_user_name,edited_good_name,edited_contact_gmail))
    conn.commit()

def logout():
    st.title("帳戶會話撤銷")
    st.write("正在登出您的帳戶。")
    if hasattr(st.session_state, "logged_in") and st.session_state.logged_in:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.clear()
        st.success("您已成功登出！")
        time.sleep(2)
        st.rerun()
    else:
        st.sidebar.info("請登錄或註冊。")



def Interact_with_My_Data_using_AI():
    st.title("使用 AI 與您的數據互動")
    st.write("向 AI 提問複雜的聚合和關鍵問題，以從用戶數據中獲取答案。")
    st.code(
        "用戶問題 - 告訴我應該在什麼日子洗衣以節省能源？ \nAI 代理 - 它將回答：根據平均太陽能節省和消耗數據，進行洗衣的最佳日子是週一和週四。")

    filter = st.selectbox("選擇要聊天的數據", ("太陽能消耗和耗散", "電力消耗", "家庭設備消耗"), index=2)
    if filter=="太陽能消耗和耗散":
        filepath=f"Data/{st.session_state.username}-solar-data.xlsx"
    elif filter=="電力消耗":
        filepath=f"Data/{st.session_state.username}-over-consumption.xlsx"
    elif filter=="家庭設備消耗":
        filepath=f"Data/{st.session_state.username}-households-data.xlsx"

    data=pd.read_excel(filepath)
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        data,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    input_text = st.text_input(f"向 AI 提問有關 {filter} 數據的任何問題")
    getans = st.button("獲取答案", use_container_width=True)
    if getans:
        result = run_chain(agent, input_text)
        st.code(result['output'])

def overall_consumption():
    overall_consumption_data['Date'] = pd.to_datetime(overall_consumption_data['Date'])
    refresh = st.button("刷新消耗儀表板", use_container_width=True)
    if refresh:
        add_random_consumption(f"Data/{st.session_state.username}-over-consumption.xlsx")
        st.rerun()

    def aggregate_consumption(df, time_period):
        # 將 'Date' 列轉換為 datetime 類型（如果尚未轉換）
        df['Date'] = pd.to_datetime(df['Date'])
        # 根據指定的時間段定義分組列
        if time_period == 'Yearly':
            group_cols = df['Date'].dt.year.rename('Year')
        elif time_period == 'Monthly':
            group_cols = [df['Date'].dt.year.rename('Year'), df['Date'].dt.month.rename('Month')]
        elif time_period == 'Weekly':
            group_cols = [df['Date'].dt.year.rename('Year'), df['Date'].dt.month.rename('Month'),
                          df['Date'].dt.isocalendar().week.rename('Week')]
        else:  # Daily
            group_cols = df['Date'].rename('Date')

        # 根據時間段進行聚合
        aggregated_df = df.groupby(group_cols).agg({
            'Daylight Consumption in kWh': 'sum',
            'Night Consumption in kWh': 'sum'
        }).reset_index()

        #
        # 可選地，在重置索引後重命名索引列以防止任何衝突
        if 'Date' in df.columns and 'Date' in aggregated_df.columns:
            aggregated_df.rename(columns={'Date': 'Date'}, inplace=True)

        if time_period == 'Yearly':
            aggregated_df['Date'] = "Year (" + aggregated_df['Year'].astype(str) + ")"
        elif time_period == 'Weekly':
            aggregated_df['Date'] = "Year (" + aggregated_df['Year'].astype(str) + ")" + "-" + "Month (" + \
                                    aggregated_df[
                                        'Month'].astype(str) + ")" + "-" + "Week (" + aggregated_df['Week'].astype(
                str) + ")"
        elif time_period == 'Monthly':
            aggregated_df['Date'] = "Year (" + aggregated_df['Year'].astype(str) + ")" + "-" + "Month (" + \
                                    aggregated_df[
                                        'Month'].astype(str) + ")"

        return aggregated_df

    def get_upcoming_dates(num_days):
        upcoming_dates = []
        today = datetime.today()
        for i in range(num_days):
            next_date = today + timedelta(days=i + 1)
            upcoming_dates.append(next_date.strftime('%Y-%m-%d'))
        return upcoming_dates

    def generate_pie_chart(df):
        # 計算每種類型的總消耗量
        total_daylight = df['Daylight Consumption in kWh'].sum()
        total_night = df['Night Consumption in kWh'].sum()

        # 創建用於餅圖的 DataFrame
        pie_df = pd.DataFrame({'Type': ['Daylight', 'Night'], 'Consumption': [total_daylight, total_night]})

        # 使用 Plotly 創建餅圖
        fig = px.pie(pie_df, names='Type', values='Consumption', title='日間與夜間消耗分佈（單位：千瓦時）')

        # 使用 Streamlit 顯示餅圖
        st.plotly_chart(fig)

    st.title("日間與夜間消耗和成本分析")
    st.write(
        "儀表板顯示日間和夜間電力消耗的數據分析和趨勢。\n\n\n\n\n.")

    dates, sliders = st.columns([1, 1])


    with dates:
        start_date = st.date_input('開始日期', overall_consumption_data['Date'].min())
        end_date = st.date_input('結束日期', overall_consumption_data['Date'].max())

    with sliders:
        daylight_min = overall_consumption_data['Daylight Consumption in kWh'].min()
        daylight_max = overall_consumption_data['Daylight Consumption in kWh'].max()
        night_min = overall_consumption_data['Night Consumption in kWh'].min()
        night_max = overall_consumption_data['Night Consumption in kWh'].max()

        daylight_range = st.slider('選擇日間消耗範圍（千瓦時）', daylight_min, daylight_max,
                                   (daylight_min, daylight_max))
        night_range = st.slider('選擇夜間消耗範圍（千瓦時）', night_min, night_max,
                                (night_min, night_max))

    col34,col33=st.columns([1,1])
    with col34:
        data_type = st.selectbox("選擇數據類型", ("日間", "夜間", "所有"), index=2)
    with col33:
        time_perioed = st.selectbox("選擇時間段類型", ("每日", "每週", "每月","每年"), index=2)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if data_type == "所有":
        day_filtered_data = overall_consumption_data[(overall_consumption_data['Date'] >= start_date) &
                                                     (overall_consumption_data['Date'] <= end_date) &
                                                     (overall_consumption_data['Daylight Consumption in kWh'] >=
                                                      daylight_range[0]) &
                                                     (overall_consumption_data['Daylight Consumption in kWh'] <=
                                                      daylight_range[1])]
        night_filtered_data = overall_consumption_data[(overall_consumption_data['Date'] >= start_date) &
                                                       (overall_consumption_data['Date'] <= end_date) &
                                                       (overall_consumption_data['Night Consumption in kWh'] >=
                                                        night_range[0]) &
                                                       (overall_consumption_data['Night Consumption in kWh'] <=
                                                        night_range[1])]
        filtered_dataa = pd.concat([day_filtered_data, night_filtered_data])
        aggregated_df = aggregate_consumption(filtered_dataa, time_perioed)

        fig = px.line(aggregated_df, x='Date', y=['Daylight Consumption in kWh', 'Night Consumption in kWh'],
                      labels={'value': '消耗量（千瓦時）'}, title='日間與夜間消耗隨時間變化的趨勢')
        fig.update_xaxes(rangeslider_visible=True)
        # 自定義佈局
        fig.update_layout(xaxis_title='日期', yaxis_title='消耗量（千瓦時）', xaxis_tickangle=-45,height=600)

        fig = px.line(aggregated_df, x='Date', y=['Daylight Consumption in kWh', 'Night Consumption in kWh'],
                      labels={'value': '消耗量（千瓦時）'}, title='日間與夜間消耗隨時間變化的趨勢')
        fig.update_xaxes(rangeslider_visible=True)
        # 自定義佈局
        fig.update_layout(xaxis_title='日期', yaxis_title='消耗量（千瓦時）', xaxis_tickangle=-45, height=600)

        # 使用 Streamlit 顯示圖表

        # 生成餅圖

        # 使用 Streamlit 顯示圖表

    elif data_type == "日間":
        filtered_dataa = overall_consumption_data[(overall_consumption_data['Date'] >= start_date) &
                                                 (overall_consumption_data['Date'] <= end_date) &
                                                 (overall_consumption_data['Daylight Consumption in kWh'] >=
                                                  daylight_range[0]) &
                                                 (overall_consumption_data['Daylight Consumption in kWh'] <=
                                                  daylight_range[1])]
        aggregated_df = aggregate_consumption(filtered_dataa, time_perioed)

        fig = px.line(aggregated_df, x='Date',
                      y='Daylight Consumption in kWh' if data_type == "日間" else 'Night Consumption in kWh',
                      labels={'value': '消耗量（千瓦時）', 'Date': '日期'},
                      title=f'{data_type} 消耗隨時間變化的趨勢')
        fig.update_xaxes(rangeslider_visible=True)

    elif data_type == "夜間":
        filtered_dataa = overall_consumption_data[(overall_consumption_data['Date'] >= start_date) &
                                                 (overall_consumption_data['Date'] <= end_date) &
                                                 (overall_consumption_data['Night Consumption in kWh'] >=
                                                  night_range[0]) &
                                                 (overall_consumption_data['Night Consumption in kWh'] <=
                                                  night_range[1])]
        aggregated_df = aggregate_consumption(filtered_dataa, time_perioed)

        fig = px.line(aggregated_df, x='Date',
                      y='Daylight Consumption in kWh' if data_type == "日間" else 'Night Consumption in kWh',
                      labels={'value': '消耗量（千瓦時）', 'Date': '日期'},
                      title=f'{data_type} 消耗隨時間變化的趨勢')
        fig.update_xaxes(rangeslider_visible=True)

    overall_consumption_data['Date'] = pd.to_datetime(overall_consumption_data['Date'])
    # 計算指標
    avg_daylight_consumption = overall_consumption_data['Daylight Consumption in kWh'].mean()
    avg_night_consumption = overall_consumption_data['Night Consumption in kWh'].mean()
    avg_daylight_price = overall_consumption_data['Daylight Price in UAH'].mean()
    avg_night_price = overall_consumption_data['Night Price in UAH'].mean()
    total_consumption = overall_consumption_data[
        ['Daylight Consumption in kWh', 'Night Consumption in kWh']].sum().sum()
    total_price = overall_consumption_data[['Daylight Price in UAH', 'Night Price in UAH']].sum().sum()
    day_night_consumption_ratio = avg_daylight_consumption / avg_night_consumption
    day_night_price_ratio = avg_daylight_price / avg_night_price
    # 計算百分比指標
    percent_daylight_consumption = (avg_daylight_consumption / total_consumption) * 100
    percent_night_consumption = (avg_night_consumption / total_consumption) * 100
    percent_daylight_price = (avg_daylight_price / total_price) * 100
    percent_night_price = (avg_night_price / total_price) * 100

    st.markdown('---')
    col1,col2,=st.columns([1,1])
    with col1:
        st.subheader("數據相關性和描述")
        st.write(filtered_dataa.drop(['Month','Year','Date'],axis=1).describe())
    with col2:
        st.subheader("篩選後的數據")
        st.write(filtered_dataa,height=90)
    # 使用 Streamlit 顯示指標
    st.title('電力消耗和價格指標\n\n.')
    col1,col2,col3,col4,col5,col6=st.columns([1,1,1,1,1,1])
    col7,col8,col9,col10,col11,col12=st.columns([1,1,1,1,1,1])

    st.title("日間成本（2.64 UAH），夜間成本（1.32 UAH）")
    with col1:
        st.metric("平均日間消耗量（千瓦時）", round(avg_daylight_consumption, 2))
    with col2:
        st.metric("平均夜間消耗量（千瓦時）", round(avg_night_consumption, 2))
    with col3:
        st.metric("平均日間價格（UAH）", round(avg_daylight_price, 2))
    with col4:
        st.metric("平均夜間價格（UAH）", round(avg_night_price, 2))
    with col5:
        st.metric("總消耗量（千瓦時）", round(total_consumption, 2))
    with col6:
        st.metric("總價格（UAH）", round(total_price, 2))
    with col7:
        st.metric("日間與夜間消耗量比率", round(day_night_consumption_ratio, 2))
    with col8:
        st.metric("日間與夜間價格比率", round(day_night_price_ratio, 2))
    with col9:
        st.metric("日間消耗量百分比",  round(percent_daylight_consumption, 2), "%")
    with col10:
        st.metric("夜間消耗量百分比",  round(percent_night_consumption, 2), "%")
    with col11:
        st.metric("日間價格百分比",  round(percent_daylight_price, 2), "%")
    with col12:
        st.metric("夜間價格百分比",  round(percent_night_price, 2), "%")
    st.markdown('---')

    col13, col14 = st.columns([1, 1])
    with col13:
        st.plotly_chart(fig, use_container_width=True)
    with col14:
        generate_pie_chart(filtered_dataa)

    # 準備建模數據

    st.markdown('---')
    # 使用 Streamlit 顯示結果
    st.title('每月能源消耗成本和預測\n\n\n.')
    number_of_month = st.slider('選擇預測的月份數', 0, 12,
                                5)

    # 創建日期索引
    overall_consumption_data['Date'] = pd.to_datetime(overall_consumption_data[['Year', 'Month']].assign(DAY=1))
    overall_consumption_data.set_index('Date', inplace=True)

    # 為 SARIMA 模型準備數據
    monthly_costs = overall_consumption_data['Total Price'].resample('M').sum()

    current_month_cost=monthly_costs[-1]

    # 排除當月數據
    current_month = monthly_costs.index[-1].strftime('%Y-%m')
    historical_costs = monthly_costs.loc[:monthly_costs.index[-2].strftime('%Y-%m')]

    order = (1, 1, 1)  # SARIMA 訂單 (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # 季節性訂單 (P, D, Q, s)
    sarima_model = sm.tsa.statespace.SARIMAX(historical_costs, order=order, seasonal_order=seasonal_order)
    fitted_sarima_model = sarima_model.fit()

    # 預測未來 5 個月
    forecast = fitted_sarima_model.forecast(steps=number_of_month)

    # 確保預測值非負
    forecast[forecast < 0] = 0

    # 提取月份名稱和年份
    month_names = forecast.index.strftime('%B')
    years = forecast.index.year

    st.metric("\n\n當月成本", f"UAH {round(current_month_cost,2)} ")
    cols = {"col1": st.columns(number_of_month)}
    # 顯示指標
    for i in range(1, number_of_month+1):
        with cols['col1'][i - 1]:
            predicted_cost = forecast[i - 1]
            previous_cost = forecast[i - 2] if i > 1 else None
            trend_color=""
            if previous_cost is not None:
                if predicted_cost > previous_cost:
                    trend_color = "%"
                elif predicted_cost < previous_cost:
                    trend_color = "-%"
                else:
                    trend_color = "%"

                st.metric(f"預測成本月份 {month_names[i - 1]} {years[i - 1]}",
                          f"UAH {predicted_cost:.2f} ",
                          trend_color)
            else:
                st.metric(f"當月預測成本 {month_names[i - 1]} {years[i - 1]}",
                          f"UAH {predicted_cost:.2f}",
                          trend_color)

    st.markdown('---')
    st.write("### 歷史數據和預測 ")
    col1,col2=st.columns([1,1])
    # 繪製歷史數據和預測
    with col1:
        fig, ax = plt.subplots()
        ax.plot(historical_costs.index, historical_costs, label='歷史每月成本')
        forecast_index = pd.date_range(start=historical_costs.index[-1], periods=number_of_month+1, freq='M')[1:]  # 下一個 5 個月
        ax.plot(forecast_index, forecast, 'ro', label='預測成本')
        ax.set_title('每月能源成本和預測')
        ax.set_xlabel('月份')
        ax.set_ylabel('總成本（UAH）')
        ax.legend()
        # 將 matplotlib 圖形轉換為 Plotly 圖形
        plotly_fig = go.Figure()
        for trace in ax.get_lines():
            x = trace.get_xdata()
            y = trace.get_ydata()
            plotly_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=trace.get_label()))

        # 將 x 軸轉換為日期格式
        plotly_fig.update_xaxes(type='date')

        # 使用 Streamlit 顯示 Plotly 圖形
        st.plotly_chart(plotly_fig)

    with col2:
        # 顯示歷史數據和預測的條形圖
        bar_chart_data = pd.DataFrame({'Date': forecast_index, 'Predicted Cost': forecast})

        # 繪製條形圖
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=month_names, y=bar_chart_data['Predicted Cost'], name='預測成本'))
        fig_bar.update_layout(barmode='group', xaxis_title='月份', yaxis_title='總成本（UAH）',
                              title='歷史數據和未來 5 個月的預測')
        st.plotly_chart(fig_bar)

    st.markdown('---')
    # 使用 Streamlit 顯示結果
    st.title('每日能源消耗成本和預測\n\n\n.')
    number_of_days = st.slider('選擇預測的天數', 0, 30, 7)

    # 重置索引以確保唯一標籤
    overall_consumption_data.reset_index(drop=True, inplace=True)

    # 排除當日數據
    historical_costs = overall_consumption_data['Total Price'][:-1]

    # 獲取歷史數據中的最後日期
    last_date = historical_costs.index[-1]

    # 為預測準備歷史數據
    historical_data = overall_consumption_data.loc[:last_date]
    # SARIMA 模型設置
    order = (1, 1, 1)  # SARIMA 訂單 (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # 季節性訂單 (P, D, Q, s)
    sarima_model = sm.tsa.statespace.SARIMAX(historical_data['Total Price'], order=order, seasonal_order=seasonal_order)
    fitted_sarima_model = sarima_model.fit()

    # 預測接下來的“number_of_days”天
    forecast = fitted_sarima_model.forecast(steps=number_of_days)
    # 確保預測值非負
    forecast[forecast < 0] = 0

    # 提取日期名稱和年份
    forecast_index = pd.date_range(start=get_upcoming_dates(len(forecast))[0], periods=number_of_days, freq='D')
    day_names = forecast_index.strftime('%A')
    st.metric(f"\n\n當日成本", f"UAH {round(float(overall_consumption_data['Total Price'].iloc[-1]), 2)}")
    cols = st.columns(number_of_days)
    # 顯示指標
    
    forecast=list(forecast)
    for i in range(number_of_days):
        predicted_cost = forecast[i]
        previous_cost = forecast[i - 1] if i > 0 else None
        trend_color = ""
        if previous_cost is not None:
            if predicted_cost > previous_cost:
                trend_color = "%"
            elif predicted_cost < previous_cost:
                trend_color = "-%"
            else:
                trend_color = "%"

            cols[i].metric(f"預測成本 第 {day_names[i]} 天",
                           f"UAH {predicted_cost:.2f}",
                           f"{trend_color}")
        else:
            cols[i].metric(f"預測成本 第 {day_names[i]} 天",
                           f"UAH {predicted_cost:.2f}",
                           f"{trend_color}")

    st.markdown('---')
    st.write("### 歷史數據和預測 ")
    col1, col2 = st.columns([1, 1])
    # 繪製歷史數據和預測
    with col1:
        fig, ax = plt.subplots()
        ax.plot(historical_costs.index, historical_costs, label='歷史每日成本')
        ax.plot(forecast_index, forecast, 'ro', label='預測成本')
        ax.set_title('每日能源成本和預測')
        ax.set_xlabel('日期')
        ax.set_ylabel('總成本（UAH）')
        ax.legend()
        # 將 matplotlib 圖形轉換為 Plotly 圖形
        plotly_fig = go.Figure()
        for trace in ax.get_lines():
            x = trace.get_xdata()
            y = trace.get_ydata()
            plotly_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=trace.get_label()))

        # 將 x 軸轉換為日期格式
        plotly_fig.update_xaxes(type='date')
        # 使用 Streamlit 顯示 Plotly 圖形
        st.plotly_chart(plotly_fig)

    with col2:
        # 顯示歷史數據和預測的條形圖
        bar_chart_data = pd.DataFrame({'Date': forecast_index, 'Predicted Cost': forecast})

        # 繪製條形圖
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=forecast_index, y=bar_chart_data['Predicted Cost'], name='預測成本'))
        fig_bar.update_layout(xaxis_title='日期', yaxis_title='總成本（UAH）',
                              title='歷史數據和未來幾天的預測')
        st.plotly_chart(fig_bar)

    st.markdown('---')
    # 使用 Streamlit 顯示結果
    st.title('家庭設備電力消耗\n\n\n.')

def house_hold_consumption():
    fig_stacked_bar=None
    def get_upcoming_dates(num_days):
        upcoming_dates = []
        today = datetime.today()
        for i in range(num_days):
            next_date = today + timedelta(days=i + 1)
            upcoming_dates.append(next_date.strftime('%Y-%m-%d'))
        return upcoming_dates
    refresh = st.button("刷新家庭設備儀表板", use_container_width=True)
    if refresh:
        add_households_consumption(f"Data/{st.session_state.username}-households-data.xlsx")
        st.rerun()
    house_holds_data['Date'] = pd.to_datetime(house_holds_data['Date'])
    filtered_data = house_holds_data

    st.title("家庭設備電力消耗")
    st.write("儀表板顯示家庭設備的數據分析。\n\n\n\n\n.")

    # 將 'Date' 列轉換為 datetime 類型
    house_holds_data['Date'] = pd.to_datetime(house_holds_data['Date'])

    # 篩選選項
    devices = house_holds_data.columns[1:-1]
    devices = ['所有設備'] + list(devices)
    selected_item = st.selectbox("選擇家庭設備", devices, index=0)
    col1,col2=st.columns([1,1])
    with col1:
        start_date = st.date_input("開始日期", house_holds_data['Date'].min())
    with col2:
        end_date = st.date_input("結束日期", house_holds_data['Date'].max())

    st.markdown('---')
    # 將日期輸入轉換為 datetime
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # 根據所選項和日期範圍篩選數據
    if selected_item == '所有設備':
        filtered_data = house_holds_data[(house_holds_data['Date'] >= start_date) & (house_holds_data['Date'] <= end_date)]
        # 顯示所有設備的使用量隨時間變化的折線圖
        fig = px.line(filtered_data, x='Date', y=filtered_data.columns[1:-1], title="所有設備的電力消耗隨時間變化")

        # 顯示所有設備總消耗量隨時間變化的條形圖
        total_consumption = filtered_data.drop(columns=['Date', 'total']).sum(axis=1)
        fig_bar = go.Figure(data=[go.Bar(x=filtered_data['Date'], y=total_consumption)])
        fig_bar.update_layout(title="所有設備的總電力消耗（千瓦時）隨時間變化", xaxis_title="日期",
                              yaxis_title="消耗量（千瓦時）")

        # 顯示各個設備的消耗量的堆積條形圖
        fig_stacked_bar = go.Figure()
        for device in devices[1:]:
            fig_stacked_bar.add_trace(go.Bar(x=filtered_data['Date'], y=filtered_data[device], name=device))

        fig_stacked_bar.update_layout(barmode='stack', title="各個設備的電力消耗（千瓦時）隨時間變化",
                                      xaxis_title="日期", yaxis_title="消耗量（千瓦時）")
    else:
        updated_items = [x for x in devices if x != "所有設備" and x!= selected_item]
        filtered_data=house_holds_data.drop(updated_items,axis=1)
        filtered_data = filtered_data[(house_holds_data['Date'] >= start_date) & (filtered_data['Date'] <= end_date)]
        # 顯示所選項的使用量隨時間變化的折線圖
        fig = px.line(filtered_data, x='Date', y=selected_item, title=f"{selected_item} 的消耗量（千瓦時）隨時間變化")
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    # 顯示其他指標
    st.metric("總能耗", f"{round(filtered_data['total'].sum(), 2)} 千瓦時")
    avg_daily_energy = filtered_data.drop('Date', axis=1).mean()
    st.subheader("平均每日能耗")
    cols = {"col1": st.columns(len(avg_daily_energy))}
    counter = 1
    # 計算每個類別的平均每日能耗
    for category, value in avg_daily_energy.items():
        with cols['col1'][counter - 1]:
            counter += 1
            st.metric(f"({category})", f"{round(value, 2)} 千瓦時")

    # 識別每日的峰值能耗
    st.markdown('---')
    filtered_data['Peak Category'] = filtered_data.drop(['Date', 'total'], axis=1).idxmax(axis=1)
    peak_counts = filtered_data['Peak Category'].value_counts()
    st.subheader("峰值能耗")
    cols = {"col1": st.columns(len(peak_counts))}
    counter = 1
    for category, count in peak_counts.items():
        with cols['col1'][counter - 1]:
            counter += 1
            st.metric(f"({category})", f"{round(value, 2)} 千瓦時")
    st.markdown('---')
    col1, col2, = st.columns([1, 1])
    with col1:
        st.subheader("數據相關性和描述")
        st.write(filtered_data.drop([ 'Date'], axis=1).describe())
    with col2:
        st.subheader("篩選後的數據")
        st.write(filtered_data, height=90)
    st.markdown('---')
    if fig_stacked_bar:
        st.plotly_chart(fig_stacked_bar, use_container_width=True)
    else:
        st.plotly_chart(fig,use_container_width=True)

    col1,col2=st.columns([1,1])
    # 顯示每個項目在總值中的比例的餅圖
    with col1:
        total_values = filtered_data.drop(columns=['Date', 'total','Peak Category']).sum(axis=0)
        fig_pie = px.pie(values=total_values, names=total_values.index, title="家庭設備的電力消耗")
        st.plotly_chart(fig_pie)

    with col2:
        # 計算每個設備的總消耗量
        device_sum =total_values
        device_sum_sorted = device_sum.sort_values(ascending=False)

        # 創建水平條形圖
        fig = go.Figure(go.Bar(
            x=device_sum_sorted.values,
            y=device_sum_sorted.index,
            orientation='h',
            text=[f'{device}: {usage} 千瓦時' for device, usage in zip(device_sum_sorted.index, device_sum_sorted.values)],
            textposition='auto'
        ))

        fig.update_layout(title="每個設備的總消耗量",
                          xaxis_title="總消耗量（千瓦時）",
                          yaxis_title="設備",
                          yaxis_categoryorder='total ascending')

        # 使用 Streamlit 顯示圖表
        st.plotly_chart(fig)

    st.markdown('---')
    # 使用 Streamlit 顯示結果
    st.title('家庭設備消耗預測\n\n\n.')
    number_of_days = st.slider('選擇預測的天數', 0, 30, 7)

    # 重置索引以確保唯一標籤
    house_holds_data.reset_index(drop=True, inplace=True)

    # 排除當日數據
    historical_costs = house_holds_data['total'][:-1]

    # 獲取歷史數據中的最後日期
    last_date = historical_costs.index[-1]

    # 為預測準備歷史數據
    historical_data = house_holds_data.loc[:last_date]
    # SARIMA 模型設置
    order = (1, 1, 1)  # SARIMA 訂單 (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # 季節性訂單 (P, D, Q, s)
    sarima_model = sm.tsa.statespace.SARIMAX(historical_data['total'], order=order, seasonal_order=seasonal_order)
    fitted_sarima_model = sarima_model.fit()
    # 預測接下來的“number_of_days”天
    forecast = fitted_sarima_model.forecast(steps=number_of_days)
    # 確保預測值非負
    forecast[forecast < 0] = 0

    # 提取日期名稱和年份
    forecast_index = pd.date_range(start=get_upcoming_dates(len(forecast))[0], periods=number_of_days, freq='D')
    day_names = forecast_index.strftime('%A')
    st.metric(f"\n\n當日設備總消耗", f"{round(float(house_holds_data['total'].iloc[-1]), 2)} 千瓦時")
    cols = st.columns(number_of_days)
    # 顯示指標
    forecast=list(forecast)
    for i in range(number_of_days):
        predicted_cost = forecast[i]
        previous_cost = forecast[i - 1] if i > 0 else None
        trend_color = ""
        if previous_cost is not None:
            if predicted_cost > previous_cost:
                trend_color = "%"
            elif predicted_cost < previous_cost:
                trend_color = "-%"
            else:
                trend_color = "%"

            cols[i].metric(f"預測（ {day_names[i]} ）",
                           f"{predicted_cost:.2f} （千瓦時）",
                           f"{trend_color}")
        else:
            cols[i].metric(f"預測（ {day_names[i]} ）",
                           f"{predicted_cost:.2f} （千瓦時）",
                           f"{trend_color}")

    st.markdown('---')
    st.write("### 歷史數據和預測 ")
    # 顯示歷史數據和預測的條形圖
    bar_chart_data = pd.DataFrame({'Date': forecast_index, 'Predicted consumpition': forecast})

    # 繪製條形圖
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=forecast_index, y=bar_chart_data['Predicted consumpition'], name='預測消耗量'))
    fig_bar.update_layout(xaxis_title='日期', yaxis_title='所有設備的總消耗量（千瓦時）',
                          title='歷史數據和未來幾天的預測')
    st.plotly_chart(fig_bar,use_container_width=True)

def solar_dacipation():
    def get_upcoming_dates(num_days):
        upcoming_dates = []
        today = datetime.today()
        for i in range(num_days):
            next_date = today + timedelta(days=i + 1)
            upcoming_dates.append(next_date.strftime('%Y-%m-%d'))
        return upcoming_dates
    refresh = st.button("刷新太陽能儀表板", use_container_width=True)
    if refresh:
        add_solar_consumption(f"Data/{st.session_state.username}-solar-data.xlsx")
        st.rerun()

    st.title("太陽能消耗和節省")
    st.write(
        "儀表板顯示太陽能發電、消耗和節省的數據分析。\n\n\n\n\n.")

    # 日期範圍篩選
    solar_data['Date'] = pd.to_datetime(solar_data['Date'])

    # 日期範圍篩選
    col1, col2 = st.columns([1, 1])
    with col1:
        start_date = st.date_input("開始日期", solar_data['Date'].min())
    with col2:
        end_date = st.date_input("結束日期", solar_data['Date'].max())

    st.markdown('---')




    # 將日期輸入轉換為 datetime
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)


filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]

# 根據所選日期範圍篩選數據
# 計算指標

average_solar_generation = filtered_data['Solar Gird Generation'].mean()
average_consumption = filtered_data['Consumption and Dicipation'].mean()
average_solar_saving = filtered_data['Solar Saving and backup'].mean()
total_solar_generation = filtered_data['Solar Gird Generation'].sum()
total_solar_saving = filtered_data['Solar Saving and backup'].sum()
total_consumption = filtered_data['Consumption and Dicipation'].sum()
peak_solar_generation = filtered_data.loc[filtered_data['Solar Gird Generation'].idxmax(), 'Solar Gird Generation']
peak_consumption = filtered_data.loc[filtered_data['Consumption and Dicipation'].idxmax(), 'Consumption and Dicipation']
peak_solar_saving = filtered_data.loc[filtered_data['Solar Saving and backup'].idxmax(), 'Solar Saving and backup']
  


    # 獲取當前日期
current_date = pd.to_datetime('today').date()

    
    
#(3/25)# 篩選當日數據  確保solar_data 變量之前進行檢查和初始化。

if solar_data is not None and house_holds_data is not None and overall_consumption_data is not None:
    # 設置 current_date
    current_date = datetime.now().date()

    # 設置 start_date 和 end_date
    start_date = solar_data['Date'].min()
    end_date = solar_data['Date'].max()

    filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]

    # 根據所選日期範圍篩選數據並計算指標
    average_solar_generation = filtered_data['Solar Gird Generation'].mean()
    average_consumption = filtered_data['Consumption and Dicipation'].mean()
    average_solar_saving = filtered_data['Solar Saving and backup'].mean()
    total_solar_generation = filtered_data['Solar Gird Generation'].sum()
    total_solar_saving = filtered_data['Solar Saving and backup'].sum()
    total_consumption = filtered_data['Consumption and Dicipation'].sum()
    peak_solar_generation = filtered_data.loc[filtered_data['Solar Gird Generation'].idxmax(), 'Solar Gird Generation']
    peak_consumption = filtered_data.loc[filtered_data['Consumption and Dicipation'].idxmax(), 'Consumption and Dicipation']
    peak_solar_saving = filtered_data.loc[filtered_data['Solar Saving and backup'].idxmax(), 'Solar Saving and backup']
  
    # 獲取當前日期
    current_date = pd.to_datetime('today').date()
    
    # 篩選當日數據
    current_day_data = solar_data[solar_data['Date'].dt.date == current_date]
    current_day_solar_generation = current_day_data['Solar Gird Generation'].sum()
    current_day_solar_diciptation = current_day_data['Consumption and Dicipation'].sum()
    current_day_solar_saving = current_day_data['Solar Saving and backup'].sum()
    col00,col0,col000,col4=st.columns([1, 1, 1, 1])
    with col0:
        st.metric("今天的太陽能發電量", f"{round(current_day_solar_generation, 2)} kWh",
                  round((average_solar_generation / average_consumption) * 100, 2))
    with col00:
        st.metric("今天的太陽能消耗量", f"{round(current_day_solar_diciptation, 2)} kWh",
                  round((average_solar_generation / average_consumption) * 100, 2))
    with col000:
        st.metric("今天的太陽能儲存量", f"{round(current_day_solar_saving, 2)} kWh",
                  round((average_solar_generation / average_consumption) * 100, 2))
    with col4:
        st.metric("總體太陽能效率", f"{round((total_solar_saving / total_solar_generation) * 100, 2) } kWh")

    st.markdown('---')
    col1, col2, col3, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1])

    with col1:
        st.metric("平均每日太陽能發電量", f"{round(average_solar_generation, 2) } kWh",
                  round((average_solar_generation / average_consumption) * 100, 2))
    with col2:
        st.metric("平均每日消耗量", f"{round(average_consumption, 2) } kWh",
                  round((average_consumption / average_solar_generation) * 100, 2))
    with col3:
        st.metric("平均每日太陽能儲存量", f"{round(average_solar_saving, 2) } kWh",
                  round((average_solar_saving / average_solar_generation) * 100, 2))

    with col5:
        st.metric("最大太陽能發電量",f"{ round(peak_solar_generation, 2) } kWh",
                  round((peak_solar_generation / total_solar_generation) * 100, 2))
    with col6:
        st.metric("最大消耗量", f"{round(peak_consumption, 2) } kWh",
                  round((peak_consumption / total_consumption) * 100, 2))
    with col7:
        st.metric("最大太陽能儲存量", f"{round(peak_solar_saving, 2) } kWh",
                  round((peak_solar_saving / total_solar_saving) * 100, 2))

    st.markdown('---')
    col1,col2,=st.columns([1,1])
    with col1:
        st.subheader("數據相關性和描述")
        st.write(filtered_data.drop(['Date'],axis=1).describe())
    with col2:
        st.subheader("過濾後的數據")
        st.write(filtered_data,height=90)

    fig3 = px.bar(filtered_data, x='Date', y=['Solar Gird Generation', 'Consumption and Dicipation', 'Solar Saving and backup'],
                  title='特定日期的太陽能發電、消耗和儲存',
                  barmode='stack')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('---')
    # 顯示結果在 Streamlit 中
    st.title('太陽能預測\n\n\n.')
    number_of_days = st.slider('選擇要預測的天數約發電量', 0, 30, 7)
    # 重置索引以確保唯一標籤
    solar_data.reset_index(drop=True, inplace=True)

    # 排除當天數據
    historical_costs = solar_data['Solar Gird Generation'][:-1]

    # 獲取歷史數據中的最後日期
    last_date = historical_costs.index[-1]

    # 為預測準備歷史數據
    historical_data = solar_data.loc[:last_date]
    # SARIMA 模型設置
    order = (1, 1, 1)  # SARIMA 順序 (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # 季節順序 (P, D, Q, s)
    sarima_model = sm.tsa.statespace.SARIMAX(historical_data['Solar Gird Generation'], order=order, seasonal_order=seasonal_order)
    fitted_sarima_model = sarima_model.fit()
    # 預測接下來的 'number_of_days' 天
    forecast = fitted_sarima_model.forecast(steps=number_of_days)
    # 確保非負預測
    forecast[forecast < 0] = 0

    # 提取日期名稱和年份
    def get_upcoming_dates(num_days):
        upcoming_dates = []
        today = datetime.today()
        for i in range(num_days):
            next_date = today + timedelta(days=i + 1)
            upcoming_dates.append(next_date.strftime('%Y-%m-%d'))
        return upcoming_dates

    forecast_index = pd.date_range(start=get_upcoming_dates(len(forecast))[0], periods=number_of_days, freq='D')
    day_names = forecast_index.strftime('%A')
    st.metric(f"\n\n當天消耗量", f"{round(float(solar_data['Solar Gird Generation'].iloc[-1]), 2)} kWh")
    cols = st.columns(number_of_days)
    # 顯示指標
    forecast=list(forecast)
    for i in range(number_of_days):
        predicted_cost = forecast[i]
        previous_cost = forecast[i - 1] if i > 0 else None
        trend_color = ""
        if previous_cost is not None:
            if predicted_cost > previous_cost:
                trend_color = "%"
            elif predicted_cost < previous_cost:
                trend_color = "-%"
            else:
                trend_color = "%"

            cols[i].metric(f"預測 ({day_names[i]})",
                           f"{predicted_cost:.2f} (kWh)",
                           f"{trend_color}")
        else:
            cols[i].metric(f"預測 ({day_names[i]})",
                           f"{predicted_cost:.2f} (kWh)",
                           f"{trend_color}")

    st.markdown('---')
    st.write("### 歷史數據和預測 ")
    # 顯示歷史數據和預測的柱狀圖
    bar_chart_data = pd.DataFrame({'Date': forecast_index, '預測發電量': forecast})

    # 繪製柱狀圖
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=forecast_index, y=bar_chart_data['預測發電量'], name='預測發電量'))
    fig_bar.update_layout(xaxis_title='日期', yaxis_title='總發電量 (kWh)',
                          title='未來幾天的歷史數據和預測')
    st.plotly_chart(fig_bar,use_container_width=True)

    st.markdown('---')

    # 在 Streamlit 應用程序頂部設置標題
    if hasattr(st.session_state, "logged_in") and st.session_state.logged_in:
        st.set_page_config(
            page_title="能源消耗分析",
            page_icon="Images/Lgo-Image-Eletric.png",  # 您可以使用表情符號或提供圖標的 URL
            layout="wide",  # 將佈局設置為寬
        )
    else:
        st.set_page_config(
            page_title="Hsieh",
            page_icon="Images/Lgo-Image-Eletric.png",  # 您可以使用表情符號或提供圖標的 URL
        )

    st.markdown("""
    <script>
    document.body.style.zoom = 0.8;
    </script>
    """, unsafe_allow_html=True)
    st.markdown("""
    <script>
    document.body.style.zoom = 0.8;
    </script>
    """, unsafe_allow_html=True)

    # 應用程序導航
    def main():
        st.sidebar.image("Images/Lgo-Image-Eletric.png", caption="能源消耗分析", use_column_width=True)

        if hasattr(st.session_state, "logged_in") and st.session_state.logged_in:
            pages = {
                "首頁": home,
                "個人資料": profile,
                "總體消耗": overall_consumption,
                "家庭消耗": house_hold_consumption,
                "太陽能消耗": solar_dacipation,
                "使用 AI 與數據互動": Interact_with_My_Data_using_AI,
                "登出帳戶": logout,
            }
            st.sidebar.title("能源分析")
            st.sidebar.markdown(f'Logged in as<span style="text-decoration:none; color:white ;">&nbsp;{st.session_state.user_good_name}</span>',unsafe_allow_html=True)
           

        else:
            pages = {
                "使用現有帳戶登錄": Login,
                "註冊新的帳戶": Register,
            }
            st.sidebar.title("帳戶")

        selection = st.sidebar.selectbox("選擇頁面導航", list(pages.keys()))
        page = pages[selection]
        page()

    if __name__ == "__main__":
        main()
else:
    st.error("數據加載失敗，請檢查文件是否存在並且格式正確。")

