import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="智慧輔具預測系統", layout="wide")

# ==========================================
# 🎨 注入 CSS：平常看舒適，但列印 PDF 時自動縮放塞進一頁
# ==========================================
st.markdown("""
    <style>
    /* 1. 隱藏預設選單、Header與浮水印 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 2. 稍微減少網頁最上方的空白 */
    .block-container {
        padding-top: 1.5rem !important;
    }

    /* 3. 針對列印(Ctrl+P)的專屬設定：強制單頁 */
    @media print {
        @page {
            size: A4 portrait;
            margin: 1cm; /* 設定 PDF 邊界 */
        }
        body {
            zoom: 0.85; /* 列印時整體自動縮小 85%，確保塞進一頁 */
        }
        /* 列印時把不需要的按鈕隱藏起來 */
        button {
            display: none !important;
        }
        /* 壓縮列印時的垂直間距 */
        div[data-testid="stVerticalBlock"] {
            gap: 0.3rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
# ==========================================

# 標題 (字體縮小版)
st.markdown("<h2 style='font-size: 22px; font-weight: bold; margin-bottom: 5px;'>🦽 臨床智慧輔具需求預測系統</h2>", unsafe_allow_html=True)
st.write("請輸入個案的基本資料與臨床代碼，系統將自動預測適合的輔具類別。")
st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)

if os.path.exists('assistive_device_15features.pkl'):
    pipeline = joblib.load('assistive_device_15features.pkl')
    imputer = pipeline['imputer']
    model = pipeline['model']
    features = pipeline['features']

    # 保持舒適的欄位寬距
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("#### 1. 基本生理數值")
        age = st.number_input("年齡", value=75.0, format="%.1f", step=0.1)
        ht = st.number_input("身高 (cm)", value=160.0, format="%.1f", step=0.1)
        wt = st.number_input("體重 (kg)", value=60.0, format="%.1f", step=0.1)

        # BMI 自動計算
        if ht > 0:
            auto_bmi = wt / ((ht / 100) ** 2)
        else:
            auto_bmi = 0.0
        bmi = st.number_input("BMI (系統自動計算)", value=float(auto_bmi), format="%.1f", disabled=True)

    with col2:
        st.markdown("#### 2. 臨床評估與身分指標")
        func_type = st.number_input("功能型態", value=1.0, format="%.1f", step=0.1)
        disability_level = st.number_input("失能等級", value=2.0, format="%.1f", step=0.1)
        category_of_care = st.number_input("照護類別", value=1.0, format="%.1f", step=0.1)
        master_placement = st.number_input("主要安置", value=0.0, format="%.1f", step=0.1)
        official_rank = st.number_input("官方等級", value=1.0, format="%.1f", step=0.1)
        address = st.number_input("居住地代碼", value=1.0, format="%.1f", step=0.1)

    with col3:
        st.markdown("#### 3. 診斷代碼 (ICD-10)")
        st.caption("※ 若無相關次要診斷，請維持 0.0")
        icd_main = st.number_input("主診斷", value=0.0, format="%.1f", step=0.1)
        icd_1 = st.number_input("次診斷 1", value=0.0, format="%.1f", step=0.1)
        icd_2 = st.number_input("次診斷 2", value=0.0, format="%.1f", step=0.1)
        icd_3 = st.number_input("次診斷 3", value=0.0, format="%.1f", step=0.1)
        icd_4 = st.number_input("次診斷 4", value=0.0, format="%.1f", step=0.1)

    st.markdown("<hr style='margin-top: 15px; margin-bottom: 15px;'>", unsafe_allow_html=True)

    # 【關鍵排版】把按鈕跟結果放在同一橫排，省下大量垂直空間！
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        # 按鈕加寬，看起來更專業
        predict_btn = st.button("🚀 執行輔具需求預測", type="primary", use_container_width=True)

    with res_col2:
        if predict_btn:
            input_dict = {
                'Category of care': category_of_care, 'Disability level': disability_level,
                'FUNC_TYPE': func_type, 'HT_y': ht, 'ICD10CM_CODE': icd_main,
                'ICD10CM_CODE_1': icd_1, 'ICD10CM_CODE_2': icd_2, 'ICD10CM_CODE_3': icd_3,
                'ICD10CM_CODE_4': icd_4, 'WT_y': wt, 'address': address,
                'age': age, 'master placement': master_placement,
                'official rank': official_rank, 'BMI': bmi
            }

            input_df = pd.DataFrame([input_dict])[features]

            try:
                input_imputed = imputer.transform(input_df)
                prediction = model.predict(input_imputed)[0]

                mapping = {
                    0: "🟢 建議評估：類別 1 (輕度輔具/單拐等)",
                    1: "🟡 建議評估：類別 2 (中度輔具/助行器等)",
                    2: "🔴 建議評估：類別 3 (重度輔具/輪椅等)"
                }

                st.success(f"**分析完成！預測結果：{mapping.get(prediction, '未知類別')}**")

            except Exception as e:
                st.error(f"預測時發生錯誤，請檢查輸入格式。詳細錯誤：{e}")
        else:
            st.info("👈 請點擊左側按鈕執行預測，結果將顯示於此。")

else:
    st.error("找不到模型檔案 `assistive_device_15features.pkl`，請確認檔案有在左側資料夾中！")
