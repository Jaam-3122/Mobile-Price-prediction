This project builds a machine learning model to classify mobile phones into four price categories — low cost, medium cost, high cost, and very high cost — based on their technical specifications. By analyzing features such as battery capacity, RAM, processor speed, camera quality, and connectivity options, the model predicts the price range of a given phone.

Objective
To develop a predictive model that:

Analyzes mobile phone specifications.

Classifies the phone into one of four price categories:

0 → Low cost

1 → Medium cost

2 → High cost

3 → Very high cost

Assists manufacturers, retailers, and customers in price estimation.

Dataset:

The dataset contains details of mobile phone specifications and their corresponding price ranges.

Features:

battery_power — Battery capacity in mAh

blue — Bluetooth availability (0/1)

clock_speed — Processor speed (GHz)

dual_sim — Dual SIM support (0/1)

fc — Front camera megapixels

four_g — 4G availability (0/1)

int_memory — Internal memory (GB)

m_dep — Mobile depth (cm)

mobile_wt — Mobile weight (grams)

n_cores — Number of processor cores

pc — Primary camera megapixels

px_height — Pixel resolution height

px_width — Pixel resolution width

ram — RAM in MB

sc_h — Screen height (cm)

sc_w — Screen width (cm)

talk_time — Battery talk time (hours)

three_g — 3G availability (0/1)

touch_screen — Touch screen availability (0/1)

wifi — WiFi availability (0/1)

Target: price_range — 0 (Low), 1 (Medium), 2 (High), 3 (Very High)

Technologies Used:

Python

Pandas, NumPy (data handling)

Matplotlib, Seaborn (visualization)

Scikit-learn (model training & evaluation)

Steps Followed:

Data Exploration

Checked data types, missing values, and feature distributions.

Visualized correlations between features and price range.

Data Preprocessing

Feature scaling using StandardScaler.

Train-test split for model evaluation.

Model Building

Trained Logistic Regression, Random Forest, and XGBoost classifiers.

Compared performance using accuracy, precision, recall, and F1-score.

Model Selection

Logistic Regression achieved 97.75% accuracy and was selected as the final model.

Model Saving

Saved the trained model for future predictions.

Results
Best Model: Logistic Regression

Accuracy: 97.75%

Key Influential Features: RAM, Pixel Resolution, Battery Power

How to Run
Clone the repository:

git clone https://github.com/Jaam-3122/Mobile-Price-prediction.git
cd mobile-price-prediction
Install dependencies:

pip install -r requirements.txt
Place dataset (dataset.csv) in the project folder.

Run training:

python train.py
Make predictions:

python predict.py --ram 4000 --px_width 1080 --px_height 1920 --battery_power 3500 ...
Future Improvements
Deploy as a web or mobile app for real-time predictions.

Use advanced ensemble models for potentially higher accuracy.

Integrate web scraping to predict prices for newly launched models.

