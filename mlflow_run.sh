#!/bin/bash

case "$1" in
    start)
        echo "🚀 Запуск MLflow..."
        mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///./mlruns/mlflow.db --default-artifact-root ./mlruns
        ;;
        
    start-d)
        echo "🚀 Запуск MLflow в фоне..."
        nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///./mlruns/mlflow.db --default-artifact-root ./mlruns > mlflow.log 2>&1 &
        echo "✅ Запущен. Логи: mlflow.log"
        ;;
        
    stop)
        echo "🛑 Остановка MLflow..."
        pkill -f "mlflow.server"
        pkill -f "gunicorn.*mlflow"
        echo "✅ Остановлен"
        ;;
        
    stop-all)
        echo "💥 Принудительная остановка..."
        pkill -9 -f "mlflow\|gunicorn.*mlflow"
        echo "✅ Все процессы остановлены"
        ;;
        
    status)
        if pgrep -f "mlflow\|gunicorn.*mlflow" > /dev/null; then
            echo "✅ MLflow запущен: http://localhost:5000"
            ps aux | grep -E "mlflow|gunicorn.*mlflow" | grep -v grep
        else
            echo "❌ MLflow не запущен"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|start-bg|stop|stop-all|status}"
        ;;
esac