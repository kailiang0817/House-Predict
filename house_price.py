# 導入需要的工具
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. 讀取房價資料
data = pd.read_csv("Taipei_house.csv")

# 2. 把我們要預測的目標（房價）和用來預測的資料（房屋條件）分開
X = data.drop(columns=["總價", "交易日期"])  # 房屋條件
y = data["總價"]  # 房價（我們要預測的）

# 3. 告訴電腦哪些欄位是文字（需要轉換），哪些是數字
text_columns = ["行政區", "車位類別"]
number_columns = [col for col in X.columns if col not in text_columns]

# 4. 準備把資料轉成電腦看得懂的樣子
preprocess = ColumnTransformer(transformers=[
    ("text", OneHotEncoder(handle_unknown="ignore"), text_columns)
], remainder="passthrough")  # 其餘數字欄位保持原樣

# 5. 做一個隨機森林模型（就像很多棵小決策樹一起做決定）
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("forest", RandomForestRegressor(random_state=0))
])

# 6. 把資料分成「訓練用」和「測試用」
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 7. 訓練模型（學習）
model.fit(X_train, y_train)

# 8. 製作互動預測！讓使用者輸入房屋條件
print("🔍 歡迎使用台北市房價預測小工具！")
print("請依照下列提示輸入房屋資訊，我們來猜猜它值多少錢 💰")

# 輸入房屋資料
district = input("👉 行政區（例如：大安區）：")
area = float(input("👉 建物總面積（平方公尺）："))
age = float(input("👉 屋齡（幾年）："))
floor = int(input("👉 樓層（例如：3）："))
total_floor = int(input("👉 總樓層（例如：10）："))
rooms = int(input("👉 房數："))
halls = int(input("👉 廳數："))
baths = int(input("👉 衛浴數："))
elevator = int(input("👉 有無電梯（有=1，無=0）："))
parking = input("👉 車位類別（例如：坡道平面、無）：")

# 整理成模型可以看懂的格式（要補齊其他欄位）
input_data = pd.DataFrame([{
    "行政區": district,
    "土地面積": 0,  # 沒輸入的用 0 補（高中生程度示意）
    "建物總面積": area,
    "屋齡": age,
    "樓層": floor,
    "總樓層": total_floor,
    "用途": 0,
    "房數": rooms,
    "廳數": halls,
    "衛數": baths,
    "電梯": elevator,
    "車位類別": parking,
    "經度": 121.5,  # 預設經緯度
    "緯度": 25.0
}])

# 預測價格
predicted_price = model.predict(input_data)[0]

# 顯示結果（加點 emoji）
print("\n📢 預測結果來囉～")
print(f"🏡 這間房子的預測總價是：{round(predicted_price)} 萬元")

if predicted_price > 6000:
    print("💎 哇！豪宅等級，適合投資！")
elif predicted_price > 3000:
    print("✨ 價格中上，位置可能不錯唷～")
else:
    print("💡 這價格滿實惠的，可以考慮看看！")
