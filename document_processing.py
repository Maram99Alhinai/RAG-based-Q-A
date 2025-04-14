from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader ,UnstructuredExcelLoader



#------------------------------ .docx 
# file_path = "data\Raw\Stats.docx"
# loader = UnstructuredWordDocumentLoader(file_path)
# documents = loader.load()

# # Print the contents
# for doc in documents:
#     print(doc.page_content)


#--------------------------------pdf
# # Path to your PDF
# pdf_path = "data\Raw\The-Alchemist.pdf"

# # Load the document
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

# # Print the text content
# for doc in documents:
#     print(doc.page_content)



#----------------------------------.xlsx

# Path to your Excel file
excel_path = "data\Raw\Loan analysis.xlsx"

# Load the spreadsheet
loader = UnstructuredExcelLoader(excel_path)
documents = loader.load()

# Print extracted content
for doc in documents:
    print(doc.page_content)