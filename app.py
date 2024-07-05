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
    try:
        plotly_code = get_plotly_code_from_gpt(df)
        html_file_path = execute_plotly_code(plotly_code, df, 0)
        return send_from_directory(app.config['static'], "enhanced_interactive_graph.html")
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
        dashboard_data = create_dashboard(summary_json, df)
        
        return jsonify({
            'message': 'Dashboard created successfully.',
            'summary': dashboard_data['summary'],
            'graphs': dashboard_data['graphs']
        })
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/bargraph', methods=['GET'])
def get_bargraph():
    return send_from_directory(app.config['static'], "bargraph.html")

@app.route('/linegraph', methods=['GET'])
def get_linegraph():
    return send_from_directory(app.config['static'], "linegraph.html")

@app.route('/piechart', methods=['GET'])
def get_piechart():
    return send_from_directory(app.config['static'], "piechart.html")

def get_plotly_code_from_gpt(dataframe, error_message=None):
    data_description = dataframe.describe(include='all').to_string()
    prompt = f"""
    Given the following data description:
    {data_description}
   
    Generate Python code using Plotly to create an enhanced interactive graph.
    The graph should be colorful, attractive, and appealing.
    It should include dynamic range colors, annotations for key points, and interactivity.
    The data contains columns representing different metrics. Use appropriate graph types based on data characteristics.
    Make sure that the generated code is not an example but actual code and also re-verify your code before generation.
   
    **Instructions**
    - Use this Dataframe in your generated code instead of giving any sample value: {dataframe}
    """
   
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

def execute_plotly_code(coder, dataframe, retries=0):
    if retries < MAX_RETRIES:
        try:
            code = coder.replace("fig.show()", "")
            local_vars = {'df': dataframe, 'px': px, 'np': np, 'make_subplots': make_subplots, 'go': go}
            exec(code, {}, local_vars)
            fig = local_vars['fig']

            static_dir = os.path.join(app.root_path, 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)

            html_file_path = os.path.join(static_dir, 'enhanced_interactive_graph.html')
            app.logger.info(f"Saving HTML file to: {html_file_path}")
            fig.write_html(html_file_path, full_html=False)

            return html_file_path
        except Exception as e:
            retries += 1
            error_message = str(e)
            print(f"Retry {retries} failed with error: {error_message}")
            coder = get_plotly_code_from_gpt(df, error_message=error_message)
            return execute_plotly_code(coder, df, retries)
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
            
            summary_lines = summary.split('\n')
            summary_json = {line.split(' - ')[0].strip(): line.split(' - ')[1].strip() for line in summary_lines}
            
            return summary_json
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise


def create_dashboard(summary, dataframe):
    fig_line = px.line(dataframe, x=dataframe.columns[0], y=dataframe.columns[1], title='Line Graph')
    fig_bar = px.bar(dataframe, x=dataframe.columns[0], y=dataframe.columns[1], title='Bar Graph')
    fig_pie = px.pie(dataframe, names=dataframe.columns[0], values=dataframe.columns[1], title='Pie Chart')

    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    bargraph_html_path = os.path.join(static_dir, 'bargraph.html')
    linegraph_html_path = os.path.join(static_dir, 'linegraph.html')
    piechart_html_path = os.path.join(static_dir, 'piechart.html')

    fig_line.write_html(linegraph_html_path, full_html=True, include_plotlyjs=False)
    fig_bar.write_html(bargraph_html_path, full_html=True, include_plotlyjs=False)
    fig_pie.write_html(piechart_html_path, full_html=True, include_plotlyjs=False)

    graph_paths = {
        'bargraph': 'bargraph.html',
        'linegraph': 'linegraph.html',
        'piechart': 'piechart.html'
    }

    summary_path = os.path.join(static_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f)

    return {
        'summary': summary_path,
        'graphs': graph_paths
    }


if __name__ == '__main__':
    app.run(debug=True)