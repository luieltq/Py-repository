#importando la libreria pandas
import pandas as pd

#importamos las funciones que usaremos del paquete selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import time

import urllib

#read_excel para leer archivos excel, openpyxl para que pueda abrir archivos xlsx
todos = pd.read_excel('./Información Cuotas NetvalU.xlsx', sheet_name= "Pago de cuotas", engine='openpyxl',header = 6, usecols="A:R")

#le quitamos el apostrofe a los numeros de celular
todos["Celular 1"] = [z[1:] for z in todos["Celular 1"]]

#deben = todos.loc[:, 'Estado'] == 'Deudor'

#deudores = todos.loc[deben]

#deudores = deudores.reset_index()

# ruta del archivo chromedriver
navegador = webdriver.Chrome(r"C:\Users\luiel\OneDrive\Documentos\chromedriver_win32\chromedriver")

# url a abrir
navegador.get("https://web.whatsapp.com/")

#tiempo de espera para que se cargue la pagina de wssp
wait = WebDriverWait(navegador, 15)

# esperando el captcha de WA
while len(navegador.find_elements_by_id("side")) < 1:
    time.sleep(5)

# preparando el loop de mensajes para todo el DataFrame seleccionado
for i, mensaje in enumerate(todos['Mensaje']):
    persona = todos.loc[i, "Miembros"]
    numero = todos.loc[i, "Celular 1"]
    texto = urllib.parse.quote(f"¡Hola, {persona}! {mensaje}")
    link = f"https://web.whatsapp.com/send?phone={numero}&text={texto}"
  # entrando al link 
    navegador.get(link)
  # esperando a que el wssp cargue
    while len(navegador.find_elements_by_id("side")) < 1:
        time.sleep(5)
  # enviando el mensaje
        wait.until(EC.visibility_of_element_located((By.XPATH, '//span[@data-testid="send"]'))).click()
    time.sleep(5)
        
    
    
    

















