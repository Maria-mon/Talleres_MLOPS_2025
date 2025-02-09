Para ejecutar en ducker:

1. construir imagen:
docker build -t logistic-api .

2. ejecutar el contenedor: 
docker run -d -p 8989:8989 logistic-api

3. Probar:
http://localhost:8989/docs

4. Click en Predict/predict, luego en try it out e ingresar los datos para las variables de entrada en los siguientes rangos:
*   culmen_length_mm: 32.1 - 59.6
*   culmen_depth_mm: 13.1 - 21.5
*   flipper_length_mm: 172 - 231
*   body_mass_g: 2700 - 6300

![image](https://github.com/user-attachments/assets/f654de8f-4af0-4516-bc51-b9710aa9ee6f)

![image](https://github.com/user-attachments/assets/bfac4808-33aa-4a59-a380-d51b329ab359)


Descripción:

En el archivo Taller_1.py Se elaboro un modelo de regresión logistica para clasificación de sexo (Male y Female) de pinguinos, basado en las siguientes variables:
*   culmen_length_mm: float
*   culmen_depth_mm: float
*   flipper_length_mm: float
*   body_mass_g: float

Se elaboro el archivo main.py para construir FastAPI.
Finalmente se elaboro los archivos DockerFile y requirements.txt para ser construir la imagen y el contenedor.


  
