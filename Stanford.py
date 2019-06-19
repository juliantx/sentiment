import stanfordnlp

config = {
	'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
	'lang': 'es', # Language code for the language to build the Pipeline in
	'tokenize_model_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora_mwt_expander.pt',
	'pos_model_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora_tagger.pt',
	'pos_pretrain_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora.pretrain.pt',
	'lemma_model_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora_lemmatizer.pt',
	'depparse_model_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora_parser.pt',
	'depparse_pretrain_path': 'C:/Users/jromanf/stanfordnlp_resources/es_ancora_models/es_ancora.pretrain.pt'
}
nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict

doc = nlp("El presidente de la Asociación de Estaciones de Servicio (bombas de gasolina), Juan Fernando Prieto Vanegas, confirmó a EL COLOMBIANO que en algunos puntos del valle de Aburrá existe un suministro de combustible, situación que, en su opinión, podría verse agravada por la el puente que viene La alerta por la falta de gasolina en las estaciones ocurrió este martes 18 de junio por la noche, cuando varios taxistas fueron al tanque y se sorprendieron de que en algunas estaciones no hubiera combustible. En otros había largas filas para abastecer. Prieto Vanegas informó que se generó la contingencia porque, según sus investigaciones, Ecopetrol suspendió el flujo a través de las tuberías debido a la emergencia causada por el movimiento de tierras en el sector de Ancón del municipio de Copacabana. La información que tengo es que pasan más de ocho días sin bombear combustible a través de este gasoducto y el Ministerio de Minas autorizó a las empresas mayoristas a abastecerse de camiones cisterna de otras plantas donde se encuentra el producto, como Sevastopol, Cartagena y otros. .. y también les autorizó a cobrar el bono de ventas , dijo el líder sindical. Agregó que la situación ha llevado al hecho de que, además de la escasez, el precio del combustible ha aumentado, aunque no especificó los porcentajes del aumento. Según su versión, los tanques de combustible se han agotado en los tanques de suministro ubicados junto a la Terminal Norte. En este momento tenemos escasez de combustible y podemos secarnos en este puente, dijo. También advirtió que hasta ahora no ha habido ningún diálogo sobre el problema con Ecopetrol. Un conductor de taxi que estaba en el tanque de bombeo de Los Buesos le dijo a Caracol Radio que mientras estaba haciendo una larga cola para obtener suministros, el gerente le informó que en esta estación el servicio de combustible ya estaba finalizando. El problema incluye estaciones ubicadas en Medellín, Sabaneta, Itagüí y otros lugares. Sin embargo, las fuentes de Ecopetrol confirmaron a EL COLOMBIANO que aunque el flujo a través del ducto que abastece al Valle de Aburrá se había cortado en Ancón, se reconectó el lunes y hoy no debería haber problemas de escasez de combustible.")

import pandas as pd

#extract lemma
def extract_lemma(doc):
    parsed_text = {'word':[], 'lemma':[]}
    for sent in doc.sentences:
       
        for wrd in sent.words:
            #extract text and lemma
            parsed_text['word'].append(wrd.text)
            
            parsed_text['lemma'].append(wrd.lemma)
    #return a dataframe
    return pd.DataFrame(parsed_text)

#call the function on doc
extract_lemma(doc)
