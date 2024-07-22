from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import openai
import os
import re
from dotenv import load_dotenv
from flask_cors import CORS
import json

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv('OPENAI_API_KEY')

app.config['static'] = 'static'  # Directory for HTML files
os.makedirs(app.config['static'], exist_ok=True)  # Ensure the HTML folder exists

# Global variable to store the dataframe
df = None

MAX_RETRIES = 4
retries = 0

@app.route('/')
def home():
    return "Welcome to the Data Analytics API! Use the /upload, /analyze, and /dashboard endpoints."

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Invalid file type, please upload a CSV, XLSX, or XLS file'}), 400

        return jsonify({'message': 'File uploaded successfully'})
    else:
        return jsonify({'error': 'Invalid file type, please upload a CSV, XLSX, or XLS file'}), 400

@app.route('/analyze', methods=['GET'])
def analyze_data():
    global df
    if df is None:
        return jsonify({'error': 'No data uploaded. Please upload a CSV file first.'}), 400
    
    user_query = request.args.get('input')

    if user_query:
        query_parts = user_query.split('*')
        if len(query_parts) > 0 and any(keyword in query_parts[0].lower() for keyword in ['chart', 'graph', 'plot', 'visualize']):
            return generate_graph(user_query)
        else:
            return generate_text_response(query_parts[0].lower())
    else:
        return jsonify({'error': 'No query provided. Please specify your query.'}), 400

def generate_text_response(query):
    global df
    try:
        # Create a concise description of the DataFrame
        dataframe = df.describe(include='all').to_string()
        
        # Modify the prompt to include the DataFrame description
        prompt = f"""
        You are a helpful assistant. Given the following data description:
        {dataframe}

        {query}
        Please provide a detailed and accurate response based on this data in proper formatting.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        text_response = response['choices'][0]['message']['content'].strip()
        return jsonify({"text_response": text_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_graph(query):
    try:
        plotly_code = get_plotly_code_from_gpt(df, query)
        html_file_path = execute_plotly_code(plotly_code, df, "enhanced_interactive_graph.html")
        return send_from_directory(app.config['static'], os.path.basename(html_file_path))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def generate_dashboard():
    global df
    if df is None:
        return jsonify({'error': 'No data uploaded. Please upload a CSV file first.'}), 400

    try:
        summary_json = get_summary_from_gpt(df)

        summary = [{"Key": k, "Value": v} for k, v in summary_json.items()]

        bar_html = get_Bargraph(df)
        line_html = get_Linechart(df)
        pie_html = get_Piechart(df)

        html_files = {
            "bar_chart": bar_html,
            "line_chart": line_html,
            "pie_chart": pie_html
        }

        return jsonify({
            "summary": summary,
            "html_files": html_files
        })
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_Bargraph(df):
    barPrompt = "Generate Python code using Plotly to create an enhanced interactive Bar graph.Graph height should be 280 px and width 394 px."
    filename = "enhanced_interactive_bar_graph.html"
    plotly_code = get_plotly_code_from_gpt(df, barPrompt)
    execute_plotly_code(plotly_code, df, filename)
    with open(os.path.join(app.config['static'], filename), 'r', encoding='utf-8') as file:
        bar_html = file.read()
    return bar_html

def get_Linechart(df):
    linePrompt = "Generate Python code using Plotly to create an enhanced interactive Line Chart.Graph height should be 280 px and width 394 px."
    filename = "enhanced_interactive_Line_chart.html"
    plotly_code = get_plotly_code_from_gpt(df, linePrompt)
    execute_plotly_code(plotly_code, df, filename)
    with open(os.path.join(app.config['static'], filename), 'r', encoding='utf-8') as file:
        line_html = file.read()
    return line_html

def get_Piechart(df):
    piePrompt = "Generate Python code using Plotly to create an enhanced interactive Pie chart.Graph height should be 280 px and width 394 px."
    filename = "enhanced_interactive_Pie_chart.html"
    plotly_code = get_plotly_code_from_gpt(df, piePrompt)
    execute_plotly_code(plotly_code, df, filename)
    with open(os.path.join(app.config['static'], filename), 'r', encoding='utf-8') as file:
        pie_html = file.read()
    return pie_html

def get_plotly_code_from_gpt(dataframe, ipPrompt, error_message=None):
    data_description = request.args.get('input')
    app.logger.info(data_description)
    prompt = f"""
    Given the following data description:
    {data_description}

    {ipPrompt}
    Generate Python code using Plotly to create an enhanced interactive.
    The graph should be colorful, attractive, and appealing.
    It should include dynamic range colors, annotations for key points, and interactivity.
    The data contains columns representing different metrics. Use appropriate graph types based on data characteristics.
    Make sure that the generated code is not an example but actual code and also re-verify your code before generation.

    **Instructions**
    - Use this Dataframe in your generated code instead of giving any sample value: {dataframe}
    """
    app.logger.info(error_message)
    if error_message:
        prompt += f"\nPrevious attempt resulted in the following error: {error_message}"

    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            code_block = response['choices'][0]['message']['content'].strip()
            print("Generated Code Block:\n", code_block)

            code_match = re.search(r"```python(.*?)```", code_block, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()

                # Ensure necessary imports
                if 'make_subplots' not in code:
                    code = "from plotly.subplots import make_subplots\n" + code
                if 'numpy' not in code:
                    code = "import numpy as np\n" + code
                if 'plotly.graph_objects as go' not in code:
                    code = "import plotly.graph_objects as go\n" + code

                return code
            else:
                raise ValueError("Code block not found in the response.")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise

def execute_plotly_code(coder, dataframe, filename, retries=0):
    if retries < MAX_RETRIES:
        try:
            code = coder.replace("fig.show()", "")
            local_vars = {'df': dataframe, 'px': px, 'np': np, 'make_subplots': make_subplots, 'go': go}
            exec(code, {}, local_vars)
            fig = local_vars['fig']

            static_dir = os.path.join(app.root_path, 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)

            html_file_path = os.path.join(static_dir, filename)
            app.logger.info(f"Saving HTML file to: {html_file_path}")
            fig.write_html(html_file_path, full_html=False)

            return html_file_path
        except Exception as e:
            retries += 1
            error_message = str(e)
            print(f"Retry {retries} failed with error: {error_message}")
            coder = get_plotly_code_from_gpt(df, ipPrompt="", error_message=error_message)
            return execute_plotly_code(coder, df, filename, retries)
    else:
        raise Exception(f"Failed after {MAX_RETRIES} retries: {error_message}")

def get_summary_from_gpt(dataframe, retries=3):
    data_description = dataframe.describe(include='all').to_string()
    prompt = f"""
    Given the following data description:
    {data_description}
    Analyze the data and provide the top 4 key summary points with actual data only. Each summary point should be presented in the following format:

    [Key Metric] - [Value]
    Ensure the response is concise and adheres strictly to the above specified format.

    **Instructions**
    - Do not give any extra text except the above specified format.
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )

            summary = response['choices'][0]['message']['content'].strip()
            print("Generated Summary:\n", summary)

            summary_dict = {}
            for line in summary.split('\n'):
                if '-' in line:
                    key, value = line.split('-', 1)
                    summary_dict[key.strip()] = value.strip()

            return summary_dict
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise

if __name__ == '__main__':
    app.run(debug=True)
