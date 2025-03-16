CUDA_VISIBLE_DEVICES=0 nohup streamlit run app.py --server.address=0.0.0.0 --server.port 8080 --server.fileWatcherType none &
cd ./web
python3 -m http.server 8000