# 导入Shiny框架的核心模块，用于构建Web应用
from shiny import App, ui, render, reactive
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入joblib库，用于加载和保存机器学习模型
import joblib
# 导入SHAP库，用于模型解释性分析
import shap
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入io模块，用于处理字节流
import io
# 导入base64模块，用于编码图像数据
import base64
# 导入pathlib模块，用于处理文件路径
from pathlib import Path
# 导入正则表达式模块，用于字符串处理
import re
# 导入warnings模块，用于控制警告信息
import warnings
# 导入nest_asyncio模块，用于在Jupyter中运行异步代码
import nest_asyncio

# 忽略所有警告信息，避免输出干扰
warnings.filterwarnings('ignore')
# 应用nest_asyncio，允许在已有事件循环中运行新的事件循环
nest_asyncio.apply()

# 设置matplotlib的中文字体，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Deja Sans']
# 设置matplotlib正确显示负号
plt.rcParams['axes.unicode_minus'] = False

# 加载标准化器
scaler = joblib.load("1.标准化数据/scaler.joblib")
selected_features = ['Aneurysm_Overlap', 'Neck_Center_Distance', 'Parent_Artery_Overlap', 'Diameter']

# 定义模型配置字典
MODEL_CONFIG = {
    "aneurysm_rf": {
        "file": "2.训练集构建模型/rf_model.joblib",
        "features": selected_features
    }
}


# 定义工具函数：清理文本以用作HTML ID
def sanitize_id(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text))


# 定义安全加载模型的函数
def safe_load_model(model_path):
    try:
        if Path(model_path).exists():
            model = joblib.load(model_path)
            return model, "Model loaded successfully"
        else:
            return None, f"Model file not found: {model_path}"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# 定义安全转换为标量的函数
def safe_convert_to_scalar(value):
    try:
        if hasattr(value, 'item'):
            return value.item()
        elif hasattr(value, '__len__') and len(value) == 1:
            return float(value[0])
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            return float(value[0])
        else:
            return float(value)
    except (ValueError, TypeError, IndexError):
        return float(value) if np.isscalar(value) else 0.0


# 定义安全格式化数值的函数
def format_value_safe(value):
    try:
        scalar_value = safe_convert_to_scalar(value)
        if abs(scalar_value) < 0.01:
            return f"{scalar_value:.4f}"
        elif abs(scalar_value) < 1:
            return f"{scalar_value:.3f}"
        elif abs(scalar_value) < 100:
            return f"{scalar_value:.2f}"
        else:
            return f"{scalar_value:.1f}"
    except Exception:
        return str(value)


# 读取背景数据
background_data_raw = pd.read_csv("1.标准化数据/train_data_scaled.csv")[selected_features]

# 定义UI
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .card { 
                border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
                margin-bottom: 20px;
            }
            .btn-primary { 
                background-color: #007bff; 
                border-color: #007bff;
                border-radius: 5px;
            }
            .form-control {
                border-radius: 5px;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
        """)
    ),

    ui.div(
        ui.h1("Intracranial Aneurysm Growth Prediction - SHAP Analysis",
              style="text-align: center; color: #2c3e50; margin-bottom: 30px; font-weight: bold;"),
        style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; margin: -15px -15px 20px -15px;"
    ),

    ui.row(
        ui.column(4,
                  ui.div(
                      ui.h3("Model Status", style="color: #2c3e50; margin-bottom: 15px;"),
                      ui.output_ui("model_status"),
                      class_="card",
                      style="padding: 20px;"
                  ),

                  ui.div(
                      ui.h3("Feature input", style="color: #2c3e50; margin-bottom: 15px;"),
                      ui.output_ui("feature_inputs"),
                      ui.br(),
                      ui.input_action_button("predict", "Start prediction",
                                             class_="btn btn-primary btn-lg",
                                             style="width: 100%; margin-top: 10px;"),
                      class_="card",
                      style="padding: 20px;"
                  )
                  ),

        ui.column(8,
                  ui.div(
                      ui.h3("Prediction Results", style="color: #2c3e50; margin-bottom: 15px;"),
                      ui.output_ui("prediction_result"),
                      class_="card",
                      style="padding: 20px; min-height: 200px;"
                  ),

                  ui.div(
                      ui.h3("SHAP Analysis", style="color: #2c3e50; margin-bottom: 15px;"),
                      ui.input_radio_buttons(
                          "plot_type",
                          "Select Chart Type:",
                          choices={
                              "shap_bar": "SHAP Bar Chart",
                              "waterfall": "Waterfall Chart",
                              "original_force": "Force Plot",
                              "custom_bar": "Custom Bar Chart"
                          },
                          selected="shap_bar",
                          inline=True
                      ),
                      ui.output_ui("shap_plot"),
                      class_="card",
                      style="padding: 20px;"
                  )
                  )
    )
)


# 定义服务器逻辑
def server(input, output, session):
    @output
    @render.ui
    def model_status():
        model_name = "aneurysm_rf"
        config = MODEL_CONFIG[model_name]
        model, status = safe_load_model(config["file"])

        if model is not None:
            status_color = "#28a745"
            status_icon = "✓"
        else:
            status_color = "#dc3545"
            status_icon = "✗"

        return ui.div(
            ui.h4(f"{status_icon} Intracranial Aneurysm Growth Random Forest Model",
                  style=f"color: {status_color}; margin-bottom: 10px;"),
            ui.p(f"Status: {status}", style="color: #666; margin-bottom: 5px;"),
            ui.p(f"Feature Count: {len(config['features'])}", style="color: #666;"),
            style="border-left: 4px solid " + status_color + "; padding-left: 15px;"
        )

    @output
    @render.ui
    def feature_inputs():
        config = MODEL_CONFIG["aneurysm_rf"]
        features = config["features"]

        inputs = []
        for feature in features:
            feature_id = sanitize_id(feature)
            inputs.append(ui.input_numeric(feature_id, f"{feature}:", value=0.0))

        return ui.div(*inputs)

    @reactive.Calc
    def load_model():
        config = MODEL_CONFIG["aneurysm_rf"]
        model, status = safe_load_model(config["file"])

        # ====================== ✅ 终极修复：彻底删除模型残留特征名 ======================
        if model is not None:
            if hasattr(model, 'feature_names_in_'):
                del model.feature_names_in_
            model.n_features_in_ = 4

        return model, config["features"], status

    @reactive.Calc
    def get_standardized_input():
        model, features, status = load_model()
        input_data = []

        for feature in features:
            feature_id = sanitize_id(feature)
            val = getattr(input, feature_id)()
            input_data.append(val)

        input_df = pd.DataFrame([input_data], columns=features)
        input_df_scaled = scaler.transform(input_df)
        input_df_scaled = pd.DataFrame(input_df_scaled, columns=features).round(2)

        return input_df, input_df_scaled

    @output
    @render.ui
    def prediction_result():
        if input.predict() == 0:
            return ui.div(
                ui.p("Please click 'Start prediction' button to make a prediction",
                     style="color: #666; text-align: center; margin-top: 50px;")
            )

        model, features, status = load_model()
        if model is None:
            return ui.div(ui.p(f"Model failed: {status}", style="color:red;"))

        try:
            input_original, input_scaled = get_standardized_input()

            # ====================== ✅ 只传数值数组，不出现特征名 ======================
            X_input = input_scaled.values

            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]

            result_text = "Aneurysm Growth" if prediction == 1 else "No Aneurysm Growth"
            confidence = max(proba) * 100
            color = "#dc3545" if prediction == 1 else "#28a745"
            bg = "#f8d7da" if prediction == 1 else "#d4edda"

            return ui.div(
                ui.h4(f"Result: {result_text}", style=f"color:{color}; font-weight:bold"),
                ui.p(f"Confidence: {confidence:.2f}%"),
                ui.p(f"No Growth Probability: {proba[0]:.4f}"),
                ui.p(f"Growth Probability: {proba[1]:.4f}"),
                ui.hr(),
                ui.h5("Your Input:"),
                ui.div(*[ui.p(f"{f}: {v}") for f, v in zip(features, input_original.iloc[0])]),
                style=f"background:{bg}; padding:20px; border-radius:8px;"
            )

        except Exception as e:
            return ui.p(f"Error: {str(e)}", style="color:red")

    @output
    @render.ui
    def shap_plot():
        if input.predict() == 0:
            return ui.p("Please predict first")

        model, features, status = load_model()
        if model is None:
            return ui.p("Model error")

        try:
            _, input_scaled = get_standardized_input()

            # ====================== ✅ 全部使用纯数组，彻底屏蔽特征名 ======================
            X_input = input_scaled.values
            bg_data = background_data_raw.values

            explainer = shap.TreeExplainer(model, bg_data)
            shap_vals = explainer.shap_values(X_input)
            base_val = explainer.expected_value

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1][0]
                base_val = base_val[1]

            plt.figure(figsize=(10, 6), dpi=120)
            exp = shap.Explanation(
                values=shap_vals, base_values=base_val,
                data=X_input[0], feature_names=features
            )
            shap.plots.bar(exp, show_data=True, show=False)
            plt.title(f'SHAP Feature Importance Analysis')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img = base64.b64encode(buf.read()).decode()
            plt.close()

            return ui.img(src=f"data:image/png;base64,{img}", style="width:100%")

        except Exception as e:
            return ui.p(f"SHAP error: {str(e)}", style="color:red")


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()