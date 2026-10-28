from src.ingest.qasper import Paragraph
from src.retrieve.embed import load_embedder, search_embed, SearchResult
from src.retrieve.rerank import load_reranker, rerank



def within_paper(paragraphs : list[Paragraph], query : str, load_embed : tuple , load_rerank : tuple , candidate_n : int = 20 , top_k: int = 5):

    e_model,e_tokenizer,e_device = load_embed
    r_model,r_tokenizer,r_device = load_rerank
    dense_list=[]
    d_list=search_embed(paragraphs,query,e_model,e_tokenizer,e_device,candidate_n,512,8)

    for ele in d_list:
        dense_list.append(ele.paragraph)

    rerank_list=rerank(query,dense_list,r_model,r_tokenizer,r_device,top_k,512,8)

    return rerank_list