from flask import Flask, request, jsonify
import joblib

# Inicialize o app Flask
app = Flask(__name__)

# Carregue o modelo salvo
modelo = joblib.load('modelo_rf.pkl')

# Defina uma rota para prever o fechamento do contrato
@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json(force=True)  # Recebe dados JSON da requisição
    previsao = modelo.predict([dados['features']])
    resultado = "Sim" if previsao[0] == 1 else "Não"
    return jsonify({"fechamento_contrato": resultado})

if __name__ == '__main__':
    app.run(debug=True)
