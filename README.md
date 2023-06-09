# Java Spring Maven Web Application

This is a Java Spring Maven web application for neural machine translation. It translates French text to English using a pre-trained model.

## Application Structure

The application consists of the following main components:

- `NeuralMachineTranslationApplication`: This class is the entry point for the application and bootstraps the Spring Boot application.

- `NeuralMachineTranslationController`: This class is a Spring MVC controller that handles the web requests. It has two endpoints: `/` for the index page and `/translate` for translating the French text.

- `NeuralMachineTranslationService`: This class is responsible for translating the French text using the neural machine translation model. It uses two pre-trained models: an encoder and a decoder.

## Dependencies

The application has the following dependencies:

- Spring Boot
- Thymeleaf
- DJL (Deep Java Library)

## Running the Application

To run the application, follow these steps:

1. Make sure you have Docker installed on your machine.
2. Build the Docker image using the provided Dockerfile: `docker build -t mdm_project .`
3. Run the Docker container: `docker run -p 8081:8081 mdm_project`

The application will be accessible at `http://localhost:8081`.

## Pre-Trained Models

The application uses pre-trained models for neural machine translation. The models are downloaded from the following URLs:

- Encoder Model: [https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_encoder_150k.zip](https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_encoder_150k.zip)
- Decoder Model: [https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_decoder_150k.zip](https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_decoder_150k.zip)

## Additional Files

The application also includes the following resource files:

- `source_wrd2idx.json`: A JSON file containing the word-to-index mapping for the French language.
- `target_idx2wrd.json`: A JSON file containing the index-to-word mapping for the English language.

These files are used by the `NeuralMachineTranslationService` to preprocess and postprocess the text for translation.

## Azure Deployment

To deploy the application to Azure using Azure Container Apps, follow these steps:

1. Install and configure the Azure CLI on your system.
2. Open your terminal.
3. Create the container app environment by running the following command:
   az containerapp env create --name neural-machine-translation --resource-group mdm-project2 --location switzerlandnorth
4. Once the environment is created, create the app and deploy it to Azure by running the following command:
   az containerapp create --name neural-machine-translation --resource-group mdm-project2 --environment neural-machine-translation --image mdm_project2-noemi/project2:latest --target-port 8081 --ingress external --query properties.configuration.ingress.fqdn

Make sure to replace `mdm-project2` with your desired resource group name.

The deployed app will be accessible using the fully qualified domain name (FQDN) returned by the deployment command.

## Credits

This application is developed as part of the MDM Project 2 at Zurich University of Applied Sciences (ZHAW).
