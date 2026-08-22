@app.get("/seiya")
def get_scalar_docs():
    return get_scalar_api_reference(app)
