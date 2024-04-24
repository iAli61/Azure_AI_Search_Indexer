# Description: This script analyzes a document with the Form Recognizer Document Analysis API utilizing the General Document Model. The results are written to a json file to files/forms_result.json
import os
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.credentials import AzureSasCredential, AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.serialization import AzureJSONEncoder
from dotenv import load_dotenv, find_dotenv
import json
from azure.storage.blob import BlobServiceClient

# loads the environment variables from the .env file
load_dotenv(find_dotenv(), override=True)

endpoint = os.getenv("document_intelligence_endpoint")
key =os.getenv("document_intelligence_key")
storage_account_name = os.getenv("storage_account_name")
sas_token = os.getenv("sas_token")
file_system_name = os.getenv("file_system_name")
directory_name = os.getenv("directory_name")
output_container_name = os.getenv("output_container_name")



def connect_to_blob_client_with_sas(account_name,file_system_name, sas_token):
    #Create a BolbServiceClient using SAS token
    blob_service_client = BlobServiceClient(account_url="{}://{}.blob.core.windows.net".format("https", storage_account_name), credential=sas_token)
    #Get a BlobClient
    blob_client = blob_service_client.get_container_client(container=file_system_name)
    return blob_client

def connect_to_adls_with_sas(account_name, file_system_name, sas_token):
    # Construct ADLS Gen2 service URL
    service_url = f"https://{account_name}.dfs.core.windows.net"
    # Create a DataLakeServiceClient using SAS token
    credential = AzureSasCredential(sas_token)
    service_client = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
            "https", storage_account_name), credential=sas_token)
    # Get a DataLakeFileSystemClient
    file_system_client = service_client.get_file_system_client(file_system=file_system_name)
    return file_system_client

    
def analyze_and_save_pdf(file_system_client,output_file_system_client, pdf_file_path, output_folder, output_container_name):
    # Set your Document Analysis endpoint and key
    document_analysis_endpoint = endpoint
    document_analysis_key = key

    # Create a DocumentAnalysisClient
    document_analysis_client = DocumentAnalysisClient(
        endpoint=document_analysis_endpoint,
        credential=AzureKeyCredential(document_analysis_key)
    )

    # Get a DataLakeFileClient for the PDF file
    file_client = file_system_client.get_file_client(pdf_file_path)
    download_stream = file_client.download_file()
    # Read the content of the PDF file
    pdf_content = download_stream.readall()

    #get tags from blob client
    # blob_client = blob_client_new.get_blob_client(blob=pdf_file_path)
    # blob_client = blob_client_new.get_container_properties()
    #tags = blob_client.get_blob_tags()

    # Get the metadata and tags of the PDF file
    metadata = file_client.get_file_properties().metadata
    
    # Analyze the PDF content
    poller = document_analysis_client.begin_analyze_document("prebuilt-document", pdf_content)
    result = poller.result().to_dict()
    
    print("Writing results to json file...")
   
    #write_json_locally(result, output_folder, pdf_file_path)
    write_to_adls(pdf_file_path, result, output_container_name, output_file_system_client, file_client, metadata)

    

def write_to_adls(pdf_file_path, result, output_container_name, output_file_system_client, file_client, metadata):
     # Get the creation date of the PDF file
    #file_properties = file_client.get_file_properties()
    #creation_date = file_properties.creation_time.strftime("%Y-%m-%d")

    # Save the result as JSON in the new container with date partitioning
    data = json.dumps(result, sort_keys=True, indent=4)

    file_client = output_file_system_client.create_file(file = pdf_file_path + ".json")
    file_client.upload_data(data=data, overwrite=True)
    file_client.set_metadata(metadata=metadata)

def write_json_locally(result, output_folder, pdf_file_path):
     # Save the result as JSON locally
   
    analyze_result_dict = result
    #write results to json file
    jsonfile = "projects/CADENT/"+ output_folder + "/"+ pdf_file_path +'.json'
    with open(jsonfile, 'w', encoding='utf-8') as f:
        
        json.dump(analyze_result_dict, f, cls=AzureJSONEncoder, ensure_ascii=False, indent=4)
    return jsonfile  

def analyze_all_pdfs_in_adls(account_name, file_system_name, sas_token, output_folder, output_container_name):
    # Connect to ADLS using SAS token
    file_system_client = connect_to_adls_with_sas(account_name, file_system_name, sas_token)
    output_file_system_client = connect_to_adls_with_sas(account_name, output_container_name, sas_token)
    # List all files in the file system
    for item in file_system_client.get_paths():
        if item.is_directory:  # Skip directories
            continue

        # Filter files with .pdf extension
        if item.name.lower().endswith('.pdf'):
            # Analyze each PDF file and save the result as JSON
            analyze_and_save_pdf(file_system_client, output_file_system_client, item.name, output_folder,output_container_name)

output_folder = "output"
if __name__ == "__main__":    
    print("Running general document analysis...")
   
    analyze_all_pdfs_in_adls(account_name=storage_account_name, file_system_name = file_system_name, sas_token = sas_token, output_folder=output_folder, output_container_name= output_container_name)
