# Supply Chain LLMOps Case Study

This project implements an AI-driven Natural Language Interface for querying supply chain data.

## Step-by-Step Instructions

1. **Create Python Virtual Environment**
Ensure you've created a python virtual environment 
```bash
python -m venv venv
```
![Create Python Environment](/docs/images/1.jpg)

2. **Activate Python virtual environment**
For Mac/ Linux enter the following command:
```bash
source venv/bin/activate
```
For Windows systems, enter the following command:
```bash
venv/Scripts/activate
```
![Create Python Environment](/docs/images/2.jpg)

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
![Install Dependencies](/docs/images/3.jpg)

4. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your OpenAI/Azure keys.

5. **Run Application**:
   ```bash
   streamlit run Src/app.py
   ```
![Run Application](/docs/images/4.jpg)

6. **Open Streamlit UI in Port 8501**:

![Streamlit UI](/docs/images/5.jpg)

![Streamlit UI](/docs/images/6.jpg)

7. **Enter the Query**:

![Query_1](/docs/images/7.jpg)

![Query_1](/docs/images/8.jpg)

8. **The RAG Application runs to provide the response**:

![Response_!](/docs/images/9.jpg)