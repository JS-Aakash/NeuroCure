@echo off
echo 🏥 AI Medical Prescription Verification System Setup
echo ============================================================

echo 🧹 Cleaning up conflicting packages...
python -m pip uninstall protobuf tensorflow transformers torch -y > nul 2>&1

echo 📦 Installing protobuf (compatible version)...
python -m pip install "protobuf==3.20.3"

echo 📦 Installing core requirements...
python -m pip install "fastapi==0.104.1"
python -m pip install "uvicorn[standard]==0.24.0"
python -m pip install "streamlit==1.28.1"
python -m pip install "openai==1.3.0"
python -m pip install "requests==2.31.0"
python -m pip install "pandas==2.1.3"
python -m pip install "numpy==1.25.2"
python -m pip install "python-multipart==0.0.6"
python -m pip install "aiohttp==3.9.1"
python -m pip install "python-dotenv==1.0.0"
python -m pip install "pydantic==2.5.0"

echo.
echo 📝 Creating .env file if it doesn't exist...
if not exist .env (
    echo # OpenRouter API Configuration > .env
    echo OPENROUTER_API_KEY=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 >> .env
    echo.
    echo # Optional: Other API keys >> .env
    echo # FDA_API_KEY=your_fda_api_key_here >> .env
    echo ✅ .env file created with OpenRouter API key.
) else (
    echo ✅ .env file already exists.
)

echo.
echo ============================================================
echo ✅ Setup completed!
echo.
echo ⚠️  The OpenRouter API key is already configured in the code
echo    If you want to use your own key, edit the .env file
echo.
echo To start the application:
echo   python start_app.py
echo.
echo Or start services individually:
echo   Backend:  python app.py
echo   Frontend: streamlit run frontend.py
echo.
pause