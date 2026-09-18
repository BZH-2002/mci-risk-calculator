import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
@st.cache_resource
def load_model():
    return joblib.load("Final_Locked_Model.joblib")

model_package = load_model()

model = model_package["fitted_estimator"]
threshold = model_package["training_derived_threshold"]
def calculate_patient_shap(
    pipeline,
    patient_data,
    model_variables,
    continuous_variables,
    categorical_variables
):
    # 取出预处理器和随机森林模型
    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["model"]

    # 按训练时相同的方法进行标准化和One-Hot编码
    transformed_data = np.asarray(
        preprocessor.transform(patient_data),
        dtype=float
    )

    # 计算SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(transformed_data)

    # 只取阳性类别（认知功能障碍 = 1）
    if isinstance(shap_values, list):
        transformed_shap = np.asarray(shap_values[1])
    else:
        transformed_shap = np.asarray(shap_values)

        if transformed_shap.ndim == 3:
            transformed_shap = transformed_shap[:, :, 1]

    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    if len(expected_value) > 1:
        base_value = float(expected_value[1])
    else:
        base_value = float(expected_value[0])

    # 建立One-Hot编码列与原始变量之间的对应关系
    mapping = list(continuous_variables)

    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    for variable, levels in zip(
        categorical_variables,
        encoder.categories_
    ):
        mapping.extend(
            [variable] * max(len(levels) - 1, 0)
        )

    mapping = np.asarray(mapping)

    # 把One-Hot后的SHAP重新聚合回原来的12个变量
    source_shap = []

    for variable in model_variables:
        value = transformed_shap[
            0,
            mapping == variable
        ].sum()

        source_shap.append(float(value))

    return np.asarray(source_shap), base_value

st.set_page_config(
    page_title="老年2型糖尿病患者3年内新发MCI风险预测",
    page_icon="🧠",
    layout="centered"
)
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 900px;
}

h1 {
    font-size: 2.1rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

div[data-testid="stVerticalBlock"] > div {
    gap: 0.6rem;
}

.stButton > button {
    width: 100%;
    height: 2.8rem;
    font-size: 1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("老年2型糖尿病患者3年内新发MCI风险预测")


age = st.number_input(
    "年龄（岁）",
    min_value=60,
    max_value=100,
    value=70,
    step=1
)

bmi = st.number_input(
    "BMI（kg/m²）",
    min_value=10.0,
    max_value=50.0,
    value=24.0,
    step=0.1
)

ssrs_total = st.number_input(
    "SSRS总分",
    min_value=0,
    max_value=100,
    value=40,
    step=1
)
depression_cn = st.selectbox(
    "抑郁",
    ["否", "是"]
)

depression_map = {
    "否": "no",
    "是": "yes"
}

depression = depression_map[depression_cn]
dpp4_cn = st.selectbox(
    "DPP-4抑制剂",
    ["否", "是"]
)

dpp4_map = {
    "否": "no",
    "是": "yes"
}

dpp4 = dpp4_map[dpp4_cn]


drink_cn = st.selectbox(
    "饮酒情况",
    ["不饮酒", "偶尔饮酒"]
)

drink_map = {
    "不饮酒": "none",
    "偶尔饮酒": "occasional"
}

drink_code_raw = drink_map[drink_cn]


exercise_cn = st.selectbox(
    "体育锻炼",
    ["规律锻炼", "不规律锻炼", "不锻炼"]
)

exercise_map = {
    "规律锻炼": "regular",
    "不规律锻炼": "irregular",
    "不锻炼": "none"
}

exercise_3cat = exercise_map[exercise_cn]


glp1_cn = st.selectbox(
    "GLP-1受体激动剂",
    ["否", "是"]
)

glp1_map = {
    "否": "no",
    "是": "yes"
}

glp1 = glp1_map[glp1_cn]


marriage_cn = st.selectbox(
    "婚姻状况",
    ["已婚", "丧偶"]
)

marriage_map = {
    "已婚": "married",
    "丧偶": "widowed"
}

marriage_code_raw = marriage_map[marriage_cn]


reading_cn = st.selectbox(
    "阅读情况",
    ["经常阅读", "偶尔阅读", "很少阅读"]
)

reading_map = {
    "经常阅读": "often",
    "偶尔阅读": "occasionally",
    "很少阅读": "rarely"
}

reading = reading_map[reading_cn]


social_cn = st.selectbox(
    "社会活动",
    ["否", "是"]
)

social_map = {
    "否": "no",
    "是": "yes"
}

social_activity = social_map[social_cn]


sulfonylurea_cn = st.selectbox(
    "磺脲类药物",
    ["否", "是"]
)

sulfonylurea_map = {
    "否": "no",
    "是": "yes"
}

sulfonylurea = sulfonylurea_map[sulfonylurea_cn]
if st.button("开始预测", type="primary", use_container_width=True):

    patient_data = pd.DataFrame([{
        "age": float(age),
        "bmi": float(bmi),
        "depression": depression,
        "dpp4": dpp4,
        "drink_code_raw": drink_code_raw,
        "exercise_3cat": exercise_3cat,
        "glp1": glp1,
        "marriage_code_raw": marriage_code_raw,
        "reading": reading,
        "social_activity": social_activity,
        "ssrs_total": float(ssrs_total),
        "sulfonylurea": sulfonylurea
    }])

    # 按原模型训练时的变量顺序排列
    patient_data = patient_data[model_package["model_variables"]]

    # 获取模型预测概率
    probabilities = model.predict_proba(patient_data)[0]

    # 明确找到阳性类别“1”对应的概率
    classes = list(model.classes_)
    positive_index = classes.index(1)

    risk = float(probabilities[positive_index])
    # ===== 个体SHAP解释 =====

    pipeline = model.best_estimator_

    source_shap, base_value = calculate_patient_shap(
        pipeline=pipeline,
        patient_data=patient_data,
        model_variables=model_package["model_variables"],
        continuous_variables=model_package["continuous_variables"],
        categorical_variables=model_package["categorical_variables"]
    )

    # 中文变量名称
    chinese_names = {
        "age": "年龄",
        "bmi": "BMI",
        "depression": "抑郁",
        "dpp4": "DPP-4抑制剂",
        "drink_code_raw": "饮酒情况",
        "exercise_3cat": "体育锻炼",
        "glp1": "GLP-1受体激动剂",
        "marriage_code_raw": "婚姻状况",
        "reading": "阅读情况",
        "social_activity": "社会活动",
        "ssrs_total": "SSRS总分",
        "sulfonylurea": "磺脲类药物"
    }

    # 当前患者的中文取值
    chinese_values = {
        "age": f"{age}岁",
        "bmi": f"{bmi:.1f} kg/m²",
        "depression": depression_cn,
        "dpp4": dpp4_cn,
        "drink_code_raw": drink_cn,
        "exercise_3cat": exercise_cn,
        "glp1": glp1_cn,
        "marriage_code_raw": marriage_cn,
        "reading": reading_cn,
        "social_activity": social_cn,
        "ssrs_total": f"{ssrs_total}分",
        "sulfonylurea": sulfonylurea_cn
    }

    feature_labels = [
        f"{chinese_names[v]} = {chinese_values[v]}"
        for v in model_package["model_variables"]
    ]




    st.divider()

    st.metric(
        "预测3年内新发MCI风险",
        f"{risk:.1%}"
    )

    if risk >= threshold:
        st.warning(
            f"模型判定：高风险\n\n"
            f"模型分类阈值：{threshold:.1%}"
        )
    else:
        st.success(
            f"模型判定：低风险\n\n"
            f"模型分类阈值：{threshold:.1%}"
        )
        st.markdown(
            f"""
            <h3 style="text-align: center;">
            模型预测老年2型糖尿病患者3年内新发MCI风险为 {risk:.1%}
            </h3>
            """,
            unsafe_allow_html=True
        )

        # 中文变量名称
        force_feature_names = [
            f"{chinese_names[v]} = {chinese_values[v]}"
            for v in model_package["model_variables"]
        ]

        # 清除之前可能残留的Matplotlib图
        plt.close("all")

        # 绘制SHAP力图
        shap.force_plot(
            base_value,
            source_shap,
            feature_names=force_feature_names,
            matplotlib=True,
            show=False,
            figsize=(12, 3),
            contribution_threshold=0.05
        )

        fig = plt.gcf()

        st.pyplot(
            fig,
            clear_figure=True,
            use_container_width=True
        )

        # 暂时用于核对SHAP计算是否与模型概率一致

