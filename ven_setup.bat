@echo off
echo 🛠️ Creating virtual environment...
python -m venv venv

echo ✅ Activating virtual environment...
call venv\Scripts\activate

echo 📦 Installing required packages...
pip install flask pandas numpy scikit-learn seaborn xgboost matplotlib 

echo 📝 Generating requirements.txt...
pip freeze > requirements.txt

echo ✅ Setup complete!
pause
