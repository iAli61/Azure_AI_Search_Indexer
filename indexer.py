import os
import json
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

import tablehelper as tb
from azure.search.documents.indexes.models import SearchIndex 
from azure.search.documents.indexes.aio import SearchIndexClient
import asyncio
import requests
import openai
from openai import AzureOpenAI 
import base64

class Indexer:
    """
    The Indexer class is responsible for creating documents, embeddings, and indexes for a given project and source.
    It utilizes the OpenAI API and FAISS vector store for generating embeddings and storing them.
    """

    def connect_to_adls_with_sas(self, output_container_name,sas_token):
        """
        Connects to Azure Data Lake Storage (ADLS) using a Shared Access Signature (SAS) token.

        Parameters:
            output_container_name (str): The name of the output container in ADLS.
            sas_token (str): The Shared Access Signature token for authentication.

        Returns:
            DataLakeFileSystemClient: A client for interacting with the specified file system in ADLS.

       
        Note:
            Ensure that the required Azure Storage and ADLS Python packages are installed.

        Raises:
            ValueError: If the `output_container_name` or `sas_token` is not provided.
        """
        service_client = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
                "https", self.storage_account_name), credential=sas_token)
        # Get a DataLakeFileSystemClient
        #file_system_client_pdf = service_client.get_file_system_client(file_system=file_system_name)
        file_system_client_json = service_client.get_file_system_client(file_system=output_container_name)
        return  file_system_client_json
    
    def write_to_adls(self, output_file_system_client, file_name, data, metadata):
        """
            Writes data to a file in Azure Data Lake Storage (ADLS).

            Parameters:
                output_file_system_client (DataLakeFileSystemClient): The client for the target file system in ADLS.
                file_name (str): The name of the file to be written in ADLS.
                data (str): The content/data to be written to the file.
                metadata (dict): A dictionary containing metadata key-value pairs to be associated with the file.

            Returns:
                None

            Note:
                Ensure that the required Azure Storage and ADLS Python packages are installed.

            Raises:
                ValueError: If the `output_file_system_client`, `file_name`, `data`, or `metadata` is not provided.
        """

        file_client = output_file_system_client.get_file_client(file_name)
        #metadata = file_client.get_file_properties().metadata
        file_client.upload_data(data, overwrite=True)
        file_client.set_metadata(metadata=metadata)

    
    def __init__(self):
        load_dotenv(find_dotenv(), override=True)
        self.endpoint = os.getenv("document_intelligence_endpoint")
        self.key =os.getenv("document_intelligence_key")
        self.storage_account_name = os.getenv("storage_account_name")
        self.sas_token = os.getenv("sas_token")
        self.file_system_name = os.getenv("file_system_name")
        self.directory_name = os.getenv("directory_name")
        self.output_container_name = os.getenv("output_container_name")
        self.historic_output_files = os.getenv("file_container_name")
        self.maxtokensize = 2048
        self.admin_key = os.getenv("admin_key")
        self.search_service_endpoint = os.getenv("search_service_endpoint")
        #print("Source: " + source)
        print("Loading JSON File")
        
        # Set OpenAI API configuration
        os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
        os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
        os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
        os.environ["OPENAI_API_TYPE"] = "azure"
        
    
    def create_docs(self, data, maxtokensize, sourcename1):
        """
        Creates documents from the given data, where each document represents a paragraph or section of the text.
        The documents are split based on the maximum token size.

        Args:
            data (dict): The data containing paragraphs and section headings.
            maxtokensize (int): The maximum number of tokens allowed in a document.
            sourcename1 (str): The name of the source.

        Returns:
            Tuple[List[Document], dict, str]: A tuple containing the created documents, page content dictionary, and full markdown text.

        Raises:
            ValueError: If a required parameter is missing or invalid.

        Note:
            Ensure that the 'Document' class is defined and imported. You can replace 'Document' with the appropriate class if needed.
         """
        # Initialize variables
        docs = []  # List to store Document objects
        pagecontent = {}  # Dictionary to store page content
        fullmdtext = ""  # String to store full markdown text
        mdtextlist = []  # List to store markdown text for each paragraph
        pagenr = []  # List to store page numbers
        pagesources = []  # List to store page sources
        sectionHeading = ""  # Variable to store section heading
        largestdoc = 0  # Variable to store the size of the largest document
        endoftable=0
        tablesearchkey=-1
        mdtext =""
        listoftext =[]

        print(" running create_docs")
            # Define a helper function to count the number of tokens in a given text
        spans={}
        #collect spans from tables
        for idx,tab in enumerate(data['tables']):
            if(len(tab['spans'])==1):
                key=tab['spans'][0]['offset']
                spans[str(key)]=idx
            else:
                smallesoffset=9999999999999999999999999
                for sp in tab['spans']:
                    if sp['offset']<smallesoffset:
                        smallesoffset=sp['offset']
                spans[str(smallesoffset)]=idx
                        
        #create pagecontent object
        pagecontent={}
        for i in range(1,len(data['pages'])+1):
            pagecontent[str(i)]=""

        # Define a helper function to count the number of tokens in a given text
        def gettokens(text):
            import tiktoken
            enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
            return len(enc.encode(text))

        # Iterate through each paragraph in the data
        #iterate over all paragraphes and create docs with mdtext seperated by sectionHeadings
        for paragraphes in data['paragraphs']:
            #when content is 7 Price conditions, then print content
            if paragraphes['spans'][0]['offset']>=endoftable:
                if 'role' in paragraphes:
                    if paragraphes['role']=='sectionHeading':
                        if sectionHeading!=paragraphes['content']:
                            #only create new doc if sectionHeading is not empty
                            if sectionHeading!='':
                                #build mdtext and create docs with content smaller than maxtokensize
                                mdtext="## "+sectionHeading+"\n\n"

                                #add content to pagecontent object
                                key=str(paragraphes['bounding_regions'][0]['page_number'])
                                if key in pagecontent:
                                    pagecontent[key]=pagecontent[key]+"## "+paragraphes['content']+"\n\n"
                                else:
                                    pagecontent[key]="## "+paragraphes['content']+"\n\n"
                                    
                                pagesources = []
                                for pid,md in enumerate(mdtextlist):
                                    if gettokens(mdtext+md)<=maxtokensize:
                                        mdtext=mdtext+md
                                        if pagenr[pid] not in pagesources:
                                            pagesources.append(pagenr[pid])
                                    else:
                                        if (gettokens(md)>maxtokensize):
                                            tokens=gettokens(md)
                                            if tokens>largestdoc:
                                                largestdoc=tokens
                                            docs.append(Document(page_content=md, metadata={"source": sourcename1, "pages":[pagenr[pid]], "tokens":tokens}))   
                                            fullmdtext=fullmdtext+md
                                        else:         
                                            tokens=gettokens(mdtext)
                                            if tokens>largestdoc:
                                                largestdoc=tokens
                                            docs.append(Document(page_content=mdtext, metadata={"source": sourcename1, "pages":pagesources, "tokens":tokens}))
                                            #add to fullmdtext
                                            fullmdtext=fullmdtext+mdtext
                                            mdtext=md
                                            pagesources = [pagenr[pid]]
                                
                                #add last doc 
                                if len(pagesources)>0:
                                    fullmdtext=fullmdtext+mdtext
                                    tokens=gettokens(mdtext)
                                    if tokens>largestdoc:
                                        largestdoc=tokens
                                    docs.append(Document(page_content=mdtext, metadata={"source": sourcename1, "pages":pagesources, "tokens":tokens}))

                                #reset mdtext and pagenr
                                mdtextlist=[]
                                pagenr=[]
                            #set new sectionHeading
                            sectionHeading=paragraphes['content']
                    else:
                        #add paragraphes to mdtext
                        mdtextlist.append(paragraphes['content']+"\n\n")
                        page=paragraphes['bounding_regions'][0]['page_number']
                        pagenr.append(page)
                        #add content to pagecontent object
                        key=str(paragraphes['bounding_regions'][0]['page_number'])
                        if key in pagecontent:
                            pagecontent[key]=pagecontent[key]+paragraphes['content']+"\n\n"
                        else:
                            pagecontent[key]=paragraphes['content']+"\n\n"

                else:
                    mdtextlist.append(paragraphes['content']+"\n\n")
                    page=paragraphes['bounding_regions'][0]['page_number']
                    pagenr.append(page)
                    #add content to pagecontent object
                    key=str(paragraphes['bounding_regions'][0]['page_number'])
                    if key in pagecontent:
                        pagecontent[key]=pagecontent[key]+paragraphes['content']+"\n\n"
                    else:
                        pagecontent[key]=paragraphes['content']+"\n\n"
                
                #add pagenr if not already in list
                page=paragraphes['bounding_regions'][0]['page_number']
                pagenr.append(page)
                                
            #add subsequent table if exists
            searchkey=str(paragraphes['spans'][0]['offset']+paragraphes['spans'][0]['length']+1)
            if tablesearchkey in spans or searchkey in spans:
                i=spans[searchkey]
                mdtextlist.append("\n\n"+tb.tabletomd(data['tables'][i])+"\n\n")
                #add content to pagecontent object
                key=str(paragraphes['bounding_regions'][0]['page_number'])
                if key in pagecontent:
                    pagecontent[key]=pagecontent[key]+"\n\n"+tb.tabletomd(data['tables'][i])+"\n\n"
                else:
                    pagecontent[key]="\n\n"+tb.tabletomd(data['tables'][i])+"\n\n"
                
                if len(data['tables'][i]['spans'])>1:
                    smallesoffset=9999999999999999999999999
                    totallength=0
                    for sp in data['tables'][i]['spans']:
                        totallength=totallength+sp['length']
                        if sp['offset']<smallesoffset:
                            key=sp['offset']
                            smallesoffset=sp['offset']
                    endoftable=smallesoffset+totallength+1
                    tablesearchkey=smallesoffset+totallength+1
                else:
                    endoftable=data['tables'][i]['spans'][0]['offset']+data['tables'][i]['spans'][0]['length']+1
                    tablesearchkey=data['tables'][i]['spans'][0]['offset']+data['tables'][i]['spans'][0]['length']+1
                page=data['tables'][i]['bounding_regions'][0]['page_number']
                pagenr.append(page)
        key=str(paragraphes['bounding_regions'][0]['page_number'])
        listoftext.append(pagecontent[key])
        for pid,md in enumerate(mdtextlist):
            if gettokens(mdtext+md)<=maxtokensize:
                mdtext=mdtext+md
                if pagenr[pid] not in pagesources:
                    pagesources.append(pagenr[pid])
            else:
                if (gettokens(md)>maxtokensize):
                    tokens=gettokens(md)
                    if tokens>largestdoc:
                        largestdoc=tokens
                    docs.append(Document(page_content=md, metadata={"source": sourcename1, "pages":[pagenr[pid]], "tokens":tokens}))   
                    fullmdtext=fullmdtext+md
                else:
                    tokens=gettokens(mdtext)
                    if tokens>largestdoc:
                        largestdoc=tokens
                    docs.append(Document(page_content=mdtext, metadata={"source": sourcename1, "pages":pagesources, "tokens":tokens}))
                    #add to fullmdtext
                    fullmdtext=fullmdtext+mdtext
                    mdtext=md
                    pagesources = [pagenr[pid]]
        
        #add last doc 
        if len(pagesources)>0:
            #add to fullmdtext
            fullmdtext=fullmdtext+mdtext
            docs.append(Document(page_content=mdtext, metadata={"source": sourcename1, "pages":pagesources, "tokens":gettokens(mdtext)}))


        print("Created "+str(len(docs))+" docs with a total of "+str(gettokens(fullmdtext))+" tokens. Largest doc has "+str(largestdoc)+" tokens.")
        return docs, pagecontent,fullmdtext
    
    print("create_docs done")
    
    async def create_embeddings(self,  chunks, jsonfilepath, historic_output_files_client, metadata, admin_key,search_service_endpoint):
        """
        Creates embeddings for the given text chunks and uploads them to an Azure Cognitive Search index.

        Args:
            chunks (List[Chunk]): List of text chunks to generate embeddings for.
            jsonfilepath (str): The path to the JSON file.
            historic_output_files_client: The DataLakeFileSystemClient for the historic output files container.
            metadata (dict): Metadata associated with the text chunks.
            admin_key (str): The admin key for the Azure Cognitive Search service.
            search_service_endpoint (str): The endpoint URL of the Azure Cognitive Search service.

        Returns:
            Tuple[SearchIndex, List[SearchIndexingResult]]: A tuple containing the SearchIndex instance and the result of the document upload.

        Raises:
            ValueError: If a required parameter is missing or invalid.

        Note:
            Ensure that the required Azure Storage, ADLS, and Azure Cognitive Search Python packages are installed.
            Make sure the chunks are instances of a class with a 'page_content' attribute.
        """
        search_service_endpoint = search_service_endpoint
        index_name = "sppi-coc-index"

        self.service_client = SearchIndexClient(search_service_endpoint,credential= AzureKeyCredential(admin_key))
        # openai.api_base = os.environ["OPENAI_API_BASE"]

        # url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

        # r = requests.get(url, headers={"api-key": os.environ["OPENAI_API_KEY"]})
        # print(chunks)
        # print(r.text)
        print("running create_embeddings")
        
        AzureOpenAIclient =AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),    api_version = "2023-05-15",
        azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
        )
        

        # embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002",model="text-embedding-ada-002", openai_api_base= openai.api_base,openai_api_type="azure",openai_api_key = os.environ["OPENAI_API_KEY"],chunk_size=16) 
        # engine ="text-embedding-ada-002"
        # Create Search Index
        fields = [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": True},
            {"name": "chunk", "type": "Edm.String", "searchable": True},
            {"name": "task_id", "type": "Edm.String", "searchable": True},
            {"name": "external_ident", "type": "Edm.String", "searchable": True},
            {"name": "quoted_on", "type": "Edm.String", "searchable": True},
            {"name": "hash_value", "type": "Edm.String", "searchable": True},
            {"name": "request_id", "type": "Edm.String", "searchable": True},
            {"name": "eval_code", "type": "Edm.String", "searchable": True},
            {"name": "md_value", "type": "Edm.String", "searchable": True},
            {"name": "vector", "type": "Collection(Edm.Double)", "searchable": False},
            # Add more fields as needed
        ]

        index_definition = SearchIndex(
            name=index_name,
            fields=fields,
            scoring_profiles=[],
            cors_options=None,
            analyzers=[],
            tokenizers=[],
            token_filters=[],
            char_filters=[],
            encryption_key=None,
            similarity=None,
                    )

        # Create or update the index
        Index_client= await self.service_client.create_or_update_index(index_definition)
        # Index documents with embeddings
       
        search_client = self.service_client.get_search_client(index_name)

        documents = []
        
       
        for i, chunk in enumerate(chunks):
            response = AzureOpenAIclient.embeddings.create(
            input= chunk.page_content,
            model="text-embedding-ada-002")
            encoded_key = base64.urlsafe_b64encode(jsonfilepath.encode()).decode()
            document = {"id": f"{encoded_key}_{i}","chunk": chunk.page_content ,"task_id": metadata.get('task_id'),"external_ident":metadata.get('external_ident') ,"quoted_on":metadata.get('quoted_on') ,"hash_value": metadata.get('hash_value'),"request_id": metadata.get('request_id'),"eval_code":metadata.get('eval_code'), "md_value":metadata.get('md_value'), "vector": response.data[0].embedding}
            documents.append(document)
          
        
        result = await search_client.upload_documents(documents)
        #print("Upload of new document succeeded: {}".format(all(x.succeeded for x in result.results)))
        self.write_to_adls(historic_output_files_client,'Documentlist/'+ jsonfilepath + '.json', json.dumps(documents, indent=4), metadata)

        return Index_client, result
    
    
    def create_index(self, jsonfilepath , maxtokensize, data, output_file_system_client, historic_output_files_client, admin_key, search_service_endpoint):
        """
        Creates an index for the given JSON file, processing the data and generating various files.

        Args:
            jsonfilepath (str): The path to the JSON file.
            maxtokensize (int): The maximum number of tokens allowed in a document.
            data (dict): The processed JSON data.
            output_file_system_client: The DataLakeFileSystemClient for the output container.
            historic_output_files_client: The DataLakeFileSystemClient for the historic output files container.
            admin_key (str): The admin key for the Azure Cognitive Search service.
            search_service_endpoint (str): The endpoint URL of the Azure Cognitive Search service.

        Returns:
            str: The path to the created JSON file.

        Raises:
            ValueError: If a required parameter is missing or invalid.

        Note:
            Ensure that the required Azure Storage, ADLS, and Azure Cognitive Search Python packages are installed.
            Make sure the output_file_system_client and historic_output_files_client are valid DataLakeFileSystemClient instances.
        """
        # Define file paths
        mdfile = jsonfilepath + '.md'
        tablemdfile = jsonfilepath + ".tables.md"
        txtfile = jsonfilepath + '.txt'
        contentjsonfile = jsonfilepath + ".pagecontent.json"
        keyvaluesjsonfile = jsonfilepath + ".keyvalues.json"
        # Remove existing files
        if os.path.exists(tablemdfile):
            os.remove(tablemdfile)
        if os.path.exists(mdfile):
            os.remove(mdfile)
        if os.path.exists(txtfile):
            os.remove(txtfile)
        if os.path.exists(contentjsonfile):
            os.remove(contentjsonfile)
        if os.path.exists(keyvaluesjsonfile):
            os.remove(keyvaluesjsonfile)

       
      
        # Load analyzer JSON file
        print("load_analyzer_json done")
        # Create documents, page content, and full markdown text
        docs, pagecontent, fullmdtext = self.create_docs(data, maxtokensize, jsonfilepath)
        print("create_docs done")
        # Write full markdown text to file
        
        mdtext = ""
        for tabid, tab in enumerate(data['tables']):
            mdtext = mdtext + "## Table " + str(tabid) + " from page " + str(tab['bounding_regions'][0]['page_number']) + "\n" + tb.tabletomd(tab) + "\n"
        
        # Generate key-value pairs
        keyvalues = {}
        for i in range(1, len(data['pages']) + 1):
            keyvalues[str(i)] = {}
        
        for keyvalue in data['key_value_pairs']:
            pagekey = str(keyvalue['key']['bounding_regions'][0]['page_number'])
            if 'value' in keyvalue:
                if keyvalue['value'] is not None:
                    keyvalues[pagekey][keyvalue['key']['content']] = keyvalue['value']['content']
        
        print("Total Nr of Docs: " + str(len(docs)))
        
        file_client = output_file_system_client.get_file_client(jsonfilepath)
        metadata = file_client.get_file_properties().metadata
        mdfile_data = fullmdtext
        self.write_to_adls(historic_output_files_client,"MDFile/" + mdfile, mdfile_data, metadata)
        
        txtfile_data = data['content']
        self.write_to_adls(historic_output_files_client,"txtFiles/" + txtfile, txtfile_data, metadata)

        contentjsonfile_data = json.dumps(pagecontent, indent=4)
        self.write_to_adls(historic_output_files_client,"PageContent/" + contentjsonfile, contentjsonfile_data, metadata)

        keyvaluesjsonfile_data = json.dumps(keyvalues, indent=4)
        self.write_to_adls(historic_output_files_client, "keyvalues_json/"+keyvaluesjsonfile, keyvaluesjsonfile_data,metadata)

        tablemdfile_data = mdtext
        self.write_to_adls(historic_output_files_client,"TableMDFile/" + tablemdfile, tablemdfile_data, metadata)
        # Rest of your code...
        # Create Indexer vector store
        print("Creating Indexer Vector Store")
        asyncio.run(self.create_embeddings( docs, jsonfilepath, historic_output_files_client, metadata, admin_key,search_service_endpoint))
        print("Done")
        
        return jsonfilepath + '.json'
    
    def load_analyzer_json(self,jsonfilepath,output_file_system_client, historic_output_files_client):
        """
        Load and process the analyzer JSON file.

        Args:
            jsonfilepath (str): The path to the JSON file.
            output_file_system_client: The DataLakeFileSystemClient for the output container.
            historic_output_files_client: The DataLakeFileSystemClient for the historic output files container.

        Returns:
            dict: The processed JSON data.

        Raises:
            FileNotFoundError: If the JSON file is not found.

        Note:
            Ensure that the required Azure Storage and ADLS Python packages are installed.
            Make sure the output_file_system_client and historic_output_files_client are valid DataLakeFileSystemClient instances.
         """
         # Implementation details...
        file_client = output_file_system_client.get_file_client(jsonfilepath)
        download_stream = file_client.download_file()
        # Read the content of the PDF file
        json_content = download_stream.readall()
        # If the content is bytes, decode it to a string
        if isinstance(json_content, bytes):
            json_content = json_content.decode('utf-8')

        # with open(json_content, encoding='utf-8') as json_file:
        data = json.loads(json_content)

        if "analyzeResult" in data:
            data = data["analyzeResult"]

        print(jsonfilepath + " loaded with " + str(len(data['pages'])) + " pages, " + str(len(data['paragraphs'])) + " paragraphs and " + str(len(data['tables'])) + " tables")

        string = json.dumps(data)
        string = string.replace("boundingRegions", "bounding_regions")
        string = string.replace("pageNumber", "page_number")
        string = string.replace("columnCount", "column_count")
        string = string.replace("rowCount", "row_count")
        string = string.replace("rowIndex", "row_index")
        string = string.replace("columnIndex", "column_index")
        string = string.replace("keyValuePairs", "key_value_pairs")

        txtfile = jsonfilepath + '.txt'
        print("Writing plain text to " + txtfile)
        metadata = file_client.get_file_properties().metadata
        # with open(txtfile, "w", encoding='utf-8') as text_file:
        #     text_file.write(data['content'])
        file_client = historic_output_files_client.create_file(file = jsonfilepath + ".txt")
        file_client.upload_data(data=data['content'], overwrite=True)
        file_client.set_metadata(metadata=metadata)
        
        

        return json.loads(string)
    
    def analyze_all_pdf_in_adls(self):
        """
        Analyzes all JSON files in Azure Data Lake Storage (ADLS) with the provided configuration.

        This method iterates through all files in the specified ADLS container, filters those with a '.json' extension,
        and analyzes each JSON file to create an index using the configured parameters.

        Parameters:
            None (Uses instance attributes for configuration parameters)

        Returns:
            None

        Note:
            Ensure that the required Azure Storage and ADLS Python packages are installed.
            Make sure the instance attributes (sas_token, output_container_name, maxtokensize, historic_output_files,
            admin_key, and search_service_endpoint) are properly set before calling this method.
        """
        # Implementation details...
        sas_token, output_container_name,maxtokensize, historic_output_files, admin_key, search_service_endpoint = self.sas_token, self.output_container_name, self.maxtokensize, self.historic_output_files, self.admin_key, self.search_service_endpoint
        # Connect to ADLS using SAS token
        #file_system_client = indexer_instance.connect_to_adls_with_sas(storage_account_name, output_container_name, sas_token)
        output_file_system_client = indexer_instance.connect_to_adls_with_sas(output_container_name ,sas_token)
        historic_output_files_client = indexer_instance.connect_to_adls_with_sas(historic_output_files ,sas_token)
        # List all files in the file system
        for item in output_file_system_client.get_paths():
            if item.is_directory:  # Skip directories
                continue

            # Filter files with .pdf extension
            if item.name.lower().endswith('.json'):
                # Analyze each PDF file and save the result as JSON
                data = indexer_instance.load_analyzer_json( item.name,output_file_system_client, historic_output_files_client)
                print("Loading Create_index", item.name)
                indexer_instance.create_index(item.name, maxtokensize, data, output_file_system_client, historic_output_files_client, admin_key, search_service_endpoint )

# Main code block
if __name__ == "__main__":

    indexer_instance = Indexer()

    indexer_instance.analyze_all_pdf_in_adls()
    # Call create_index method to create the index
