# CommVulnHunter

## commande to build the image
```bash
docker build -t phishing-detector . 
```
## commande to run the image
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 80
```