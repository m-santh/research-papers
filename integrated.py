import gradio as gr
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import re

# Load the semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

# Mapping for conferences
A_STAR_CONFERENCES = {
    "ISCA": "ACM International Symposium on Computer Architecture",
    "SOSP": "ACM SIGOPS Symposium on Operating Systems Principles",
    "PODC": "ACM Symposium on Principles of Distributed Computing",
    "ASPLOS": "Architectural Support for Programming Languages and Operating Systems",
    "HPCA": "International Symposium on High Performance Computer Architecture",
    "OSDI": "Usenix Symposium on Operating Systems Design and Implementation"
}
A_CONFERENCES = {
    "HPDC": "ACM International Symposium on High Performance Distributed Computing",
    "Middleware": "ACM/IFIP/USENIX International Middleware Conference",
    "FAST": "Conference on File and Storage Technologies",
    "ICS": "International Conference on Supercomputing",
    "EuroSys": "Eurosys Conference",
    "IPDPS": "IEEE International Parallel and Distributed Processing Symposium",
    "SC": "International Conference for High Performance Computing, Networking, Storage and Analysis",
    "ICDCS": "International Conference on Distributed Computing Systems",
    "DISC": "International Symposium on Distributed Computing",
    "USENIX": "Usenix Annual Technical Conference",
    "HotOS": "USENIX Workshop on Hot Topics in Operating Systems",
}

def get_abstract_from_doi(doi_url):
    if not doi_url:
        return ""
    try:
        resp = requests.get(doi_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        if "acm.org" in doi_url:
            abstract_div = soup.find("div", class_="abstractSection")
        elif "ieeexplore.ieee.org" in doi_url:
            abstract_div = soup.find("div", class_="abstract-text")
            if not abstract_div:
                abstract_div = soup.find("meta", {"name": "description"})
                return abstract_div['content'] if abstract_div else None
        elif "springer.com" in doi_url:
            abstract_div = soup.find("section", class_="Abstract")
            if not abstract_div:
                abstract_div = soup.find("meta", {"name": "dc.Description"})
                return abstract_div['content'] if abstract_div else None
        elif "sciencedirect.com" in doi_url:
            abstract_div = soup.find("div", class_="abstract author")
        else:
            return "Unsupported DOI domain"

        return abstract_div.text.strip() if abstract_div else "Abstract not found"
    except Exception as e:
        return f"Error: {str(e)}"

def semantic_filter(text, ref_embedding, authors=None, threshold=0.4):
    filtered = False
    author_count = Counter()

    if not text:
        return filtered, author_count

    title_embedding = model.encode(text, convert_to_tensor=True)
    similarity = float(util.cos_sim(title_embedding, ref_embedding)[0][0])

    if similarity > threshold:
        filtered = True
        if authors:
            for author in authors:
                author_count[author] += 1

    return (similarity if filtered else 0.0), author_count

def fetch_dblp_entries(conference_key, year, ref_embedding):
    base_url = f"https://dblp.org/search?q=streamid:conf/{conference_key}: year:{year}:"
    try:
        res = requests.get(base_url)
        res.raise_for_status()
    except requests.RequestException:
        return ([], [], [])

    try:
        soup = BeautifulSoup(res.content, "html.parser")
        papers = soup.find_all("cite", class_="data")
        
        results = []
        filtered = []
        threshold = 0.4
        author_stats = Counter()

        for paper in papers:
            title_tag = paper.find("span", class_="title")
            if not title_tag:
                continue
            title = title_tag.get_text()
            authors = [a.get_text() for a in paper.find_all("span", itemprop="author")]
            links = paper.find_all("a", href=True)
            entry_li = paper.find_parent("li", class_="entry")
            ee_li = entry_li.find("li", class_="ee") if entry_li else None

            #ee_li = paper.find("li", class_="ee")
            link = ee_li.a["href"] if ee_li and ee_li.a else None
            #link = next((l["href"] for l in links if "doi" in l["href"] or "pdf" in l["href"]), None)
            abstract = get_abstract_from_doi(link)
            full_text = title + " " + abstract if abstract else title
            score, author_count = semantic_filter(full_text, ref_embedding, authors)
            score = round(score, 2)
            print(paper)
            print(ee_li)

            results.append((title, authors, link or "", score, year, abstract or ""))

            if score > threshold:
                filtered.append((title, authors, link or "", score, year, abstract or ""))
                author_stats.update(author_count)

        return (results, filtered, author_stats)
    except Exception as e:
        print(f"Error parsing DBLP results: {e}")
        return ([], [], [])

def run_stream(a_star_input, a_input, start_year, end_year, query, sort_option): #, progress=gr.Progress(track_tqdm=True)):
    confs_input = (a_star_input or []) + (a_input or [])
    years = list(range(start_year, end_year + 1))
    total = len(confs_input) * len(years)

    if not confs_input:
        yield "No valid conferences selected.", "", "", "", 0
        return

    all_papers = []
    all_filtered = []
    all_authors = Counter()
    yield "Fetching papers...", "", "", "", "", 0

    ref_embedding = model.encode(query, convert_to_tensor=True)

    count = 0
    count_progress = 0
    for conf in confs_input:
        for year in years:
            count += 1
            new_papers, filtered_paper, authors = fetch_dblp_entries(conf.lower(), year, ref_embedding)
            all_papers.extend(new_papers)
            all_filtered.extend(filtered_paper)
            all_authors.update(authors)

            paper_status = "\n".join([paper[0] for paper in all_papers])

            count_progress = round((count * 100) / total, 2)

            # Sort on-the-fly for live display (optional)
            if sort_option == "Most Recent First":
                all_filtered = sorted(all_filtered, key=lambda x: -int(x[5]))
                all_papers = sorted(all_papers, key=lambda x: -int(x[5]))

            elif sort_option == "Oldest First":
                all_filtered = sorted(all_filtered, key=lambda x: int(x[5]))
                all_papers = sorted(all_papers, key=lambda x: int(x[5]))
            elif sort_option == "Relevance":
                all_filtered = sorted(all_filtered, key=lambda x: -float(x[3]))
                all_papers = sorted(all_papers, key=lambda x: -float(x[3]))

            paper_display = "\n\n".join([
                f"""{year} | Score: {score:.2f}<br>
               <b>Title:</b> {title}<br>
               <b>Authors:</b> {', '.join(authors)}<br>
               <b>Link:</b> <a href="{link}" target="_blank">{link}</a><br>"""
               for title, authors, link, score, year, abstract in all_filtered
            ])

            all_paper_display = "\n\n".join([
                f"""{year} | Score: {score:.2f}<br>
               <b>Title:</b> {title}<br>
               <b>Authors:</b> {', '.join(authors)}<br>
               <b>Link:</b> <a href="{link}" target="_blank">{link}</a><br>"""
               for title, authors, link, score, year, abstract in all_filtered
            ])

            author_display = "\n".join([
                f"{author}: {count} papers" for author, count in all_authors.most_common(20)
            ])


            yield f"Fetched {len(all_papers)} papers so far...", \
                  f"Matched {len(all_filtered)} papers so far...", \
                  paper_display, author_display, all_paper_display, count_progress

    if sort_option == "Most Recent First":
        all_filtered = sorted(all_filtered, key=lambda x: -int(x[5]))
        all_papers = sorted(all_papers, key=lambda x: -int(x[5]))

    elif sort_option == "Oldest First":
        all_filtered = sorted(all_filtered, key=lambda x: int(x[5]))
        all_papers = sorted(all_papers, key=lambda x: int(x[5]))

    elif sort_option == "Relevance":
        all_filtered = sorted(all_filtered, key=lambda x: -int(x[3]))
        all_papers = sorted(all_papers, key=lambda x: -float(x[3]))

    paper_display = "\n\n".join([
                f"""{year} | Score: {score:.2f}<br>
               <b>Title:</b> {title}<br>
               <b>Authors:</b> {', '.join(authors)}<br>
               <b>Link:</b> <a href="{link}" target="_blank">{link}</a><br>"""
               for title, authors, link, score, year, abstract in all_filtered
    ])

    all_paper_display = "\n\n".join([
                f"""{year} | Score: {score:.2f}<br>
               <b>Title:</b> {title}<br>
               <b>Authors:</b> {', '.join(authors)}<br>
               <b>Link:</b> <a href="{link}" target="_blank">{link}</a><br>"""
               for title, authors, link, score, year, abstract in all_filtered
    ])

    author_display = "\n".join([
        f"{author}: {count} papers" for author, count in all_authors.most_common(20)
    ])

    yield f"Total papers retrieved: {len(all_papers)}", \
          f"Total matching papers: {len(all_filtered)}", \
          paper_display, author_display, all_paper_display, count_progress


def format_papers(paper_list):
    return "\n\n".join([
        f"{year} | Score: {score:.2f}\nTitle: {title}\nAuthors: {', '.join(authors)}\nLink: {link}"
        for title, authors, link, score, year, abstract in paper_list
    ])

def format_authors(author_counter):
    return "\n".join([
        f"{author}: {count} papers" for author, count in author_counter.most_common(20)
    ])

# Gradio UI
def launch_gui():
    with gr.Blocks() as demo:
        gr.Markdown("""# Semantic Scholar-like DBLP Explorer""")

        with gr.Row(): 
            with gr.Column():
                a_star_input = gr.CheckboxGroup(
                    choices=list(A_STAR_CONFERENCES.keys()),  # Replace with actual A* conferences
                    label="A* Conferences"
               )
            with gr.Column():
                a_input = gr.CheckboxGroup(
                    choices=list(A_CONFERENCES.keys()),  # Replace with actual A conferences
                    label="A Conferences"
                )

        with gr.Row(): 
            start_year = gr.Number(label="Start Year", value=2020, precision=0)
            end_year = gr.Number(label="End Year", value=2025, precision=0)

        query = gr.Textbox(label="Search Keywords (e.g., Vision Transformer)", value="Systems engineering and GPU")
        sort_option = gr.Dropdown(["Relevance", "Most Recent First", "Oldest First"], label="Sort by", value="Relevance")

        run_btn = gr.Button("Run Search")

        #total_count = gr.Number(value=10, visible=False) 
        progress_count = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Progress (%)", interactive=False)
        with gr.Row():
            summary_output = gr.Textbox(label="Total paper summary", lines=1)
            summary_match_output = gr.Textbox(label="Total matched summary", lines=1)

        with gr.Row():
            matching_papers_output = gr.Textbox(label="Matching Papers", lines=10)
            top_authors_output = gr.Textbox(label="Top Authors", lines=10)
        
        all_papers_output = gr.Textbox(label="All Fetched Paper Titles", lines=10)

        #def update_slider_max(total, progress):
        #    return gr.update(maximum=total, value=progress)

        #run_btn.click(fn=update_slider_max, inputs=[total_count, progress_count], outputs=progress_count)
        run_btn.click(fn=run_stream,
                      inputs=[a_star_input, a_input, start_year, end_year, query, sort_option],
                      outputs=[summary_output, summary_match_output, matching_papers_output, top_authors_output, all_papers_output, progress_count])

    demo.launch()

if __name__ == '__main__':
    launch_gui()
