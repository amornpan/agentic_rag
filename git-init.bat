@echo off
echo Initializing Git repository...
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/amornpan/agentic_rag.git
echo.
echo Git repository initialized successfully.
echo.
echo To push to GitHub, run the following command:
echo git push -u origin main
echo.
echo Note: You might need to authenticate with GitHub if you haven't done so already.
