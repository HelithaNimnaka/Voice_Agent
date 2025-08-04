@echo off
echo 🏦 Starting LB Finance Banking Chatbot...
echo 🌐 The app will open in your web browser shortly...
echo 📱 Access URL: http://localhost:8501
echo ⚡ Press Ctrl+C to stop the server
echo.

streamlit run streamlit_banking_app.py --server.port 8501 --server.address localhost
