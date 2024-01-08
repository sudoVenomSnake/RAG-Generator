import tempfile
import os
import zipfile
import json

from llama_index import SimpleDirectoryReader, ServiceContext, set_global_service_context, get_response_synthesizer, TreeIndex, VectorStoreIndex
from llama_index.indices.tree import TreeRootRetriever
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms import OpenAI
import tiktoken

import streamlit as st

st.set_page_config(layout = "wide")

st.title("Quick RAG App Generator")

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "index" not in st.session_state:
    st.session_state.index = False

def file_uploaded():
    st.session_state.file_uploaded = True
    st.session_state.index = False
    st.session_state.file_names = []
    st.session_state.summaries = []

def downloaded():
    st.session_state.file_uploaded = False
    st.session_state.index = False
    os.remove("app.zip")

with st.expander(label = "File Upload", expanded = not st.session_state.file_uploaded):
    with st.form("file-upload"):
        files = st.file_uploader(label = "Please upload the files needed in the RAG App -", accept_multiple_files = True)
        st.session_state.title = st.text_input(label = "Please provide a title to the app -", value = "")
        st.session_state.model_choice = st.selectbox("Which OpenAI model do you want to use?", options = ["gpt-4", "gpt-3.5-turbo"], index = 1)
        submit = st.form_submit_button(label = "Submit", on_click = file_uploaded)

if st.session_state.file_uploaded and not st.session_state.index:
    loaded_files = []
    chatgpt = OpenAI(temperature = 0, model = st.session_state.model_choice)
    token_counter = TokenCountingHandler(tokenizer = tiktoken.encoding_for_model(st.session_state.model_choice).encode)
    callback_manager = CallbackManager([token_counter])
    service_context = ServiceContext.from_defaults(llm = chatgpt, callback_manager = callback_manager, chunk_size = 512)
    set_global_service_context(service_context)

    my_bar = st.progress(0, text = "Building Index")
    for n, file in enumerate(files):
        temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(temp_dir.name, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        docs = SimpleDirectoryReader(input_files = [file_path]).load_data()
        st.write(f"{file.name} - Loaded {len(docs)} Pages.")
        index = TreeIndex.from_documents(docs, show_progress = True)
        st.session_state.summaries.append(TreeRootRetriever(index).retrieve("")[0].text)
        print(f"Built {len(index.docstore.docs)} Nodes.")
        index.storage_context.persist(persist_dir = file.name)
        st.session_state.file_names.append(file.name)
        my_bar.progress(int((n + 1) / len(files) * 100), text = "Building Index")
    
    st.session_state.index = True

    for i, j in zip(st.session_state.file_names, st.session_state.summaries):
        st.subheader(i)
        st.write(j)

    # _ = st.button("Proceed to download")

    # st.markdown(f"**Embedding Tokens:** {token_counter.total_embedding_token_count}")
    # st.markdown(f"**LLM Prompt Tokens:** {token_counter.prompt_llm_token_count}")
    # st.markdown(f"**LLM Completion Tokens:** {token_counter.completion_llm_token_count}")
    # st.markdown(f"**Total LLM Token Count:** {token_counter.total_llm_token_count}")

    st.write(f"{((token_counter.prompt_llm_token_count / 1000) * 0.01) + (token_counter.completion_llm_token_count / 1000) * 0.03} $ Used." if st.session_state.model_choice == "gpt-4" else f"{((token_counter.prompt_llm_token_count / 1000) * 0.001) + (token_counter.completion_llm_token_count / 1000) * 0.002} $ Used.")

app_code = """import json
import os
import redirect as rd

from llama_index import StorageContext, load_index_from_storage, ServiceContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.query_engine import SubQuestionQueryEngine

import streamlit as st

st.set_page_config(layout = "wide")

if "key" not in st.session_state:
    os.environ["OPENAI_API_KEY"] = st.text_input(label = "Please enter your OpenAI key")
    if os.environ["OPENAI_API_KEY"]:
        st.session_state.key = True

if "loaded" not in st.session_state:
    with open("template.json", "r") as f:
        template = json.load(f)
    st.session_state.title = template["title"]
    st.session_state.index_files = template["index_names"]
    st.session_state.summaries = template["summaries"]
    st.session_state.model_choice = template["model_choice"]
    st.session_state.loaded = True
    
@st.cache_resource
def initialize():
    query_engine_tools = []
    for i, j in zip(st.session_state.index_files, st.session_state.summaries):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir = "RAG Generator/" + i), service_context = ServiceContext.from_defaults(llm = OpenAI(temperature = 0, model = st.session_state.model_choice)))
        query_engine_tools.append(QueryEngineTool(
            query_engine = index.as_query_engine(similarity_top_k = 3),
            metadata = ToolMetadata(
                name = i,
                description = (j + " ",
                    "Properly frame question as it will match relevance to compute answer."
                ),
            ),
        ))
    llm = OpenAI(model = st.session_state.model_choice)
    query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools = query_engine_tools, verbose = True)
    final_tool = [QueryEngineTool(
            query_engine = query_engine,
            metadata = ToolMetadata(
                name = "Index",
                description = ("This tool is the Master Index and the source for your answers. ",
                    "Add as much information about the query as possible so the tool excels."
                ),
            ),
        )]
    st.session_state.agent = ReActAgent.from_tools(final_tool, llm = llm, verbose = True)
    return True

if "key" in st.session_state:
    initialize()
    st.title(st.session_state.title)
    for i, j in zip(st.session_state.index_files, st.session_state.summaries):
        st.subheader(i)
        st.write(j)
    query = st.text_input(label = "Please enter your query -")
    if query:
        with rd.stdout as out:
            st.session_state.agent.query(query)"""
redirect_code = """import streamlit as st
import io
import contextlib
import sys
import re


class _Redirect:
    class IOStuff(io.StringIO):
        def __init__(self, trigger, max_buffer, buffer_separator, regex, dup=None):
            super().__init__()
            self._trigger = trigger
            self._max_buffer = max_buffer
            self._buffer_separator = buffer_separator
            self._regex = regex and re.compile(regex)
            self._dup = dup

        def write(self, __s: str) -> int:
            if self._max_buffer:
                concatenated_len = super().tell() + len(__s)
                if concatenated_len > self._max_buffer:
                    rest = self.get_filtered_output()[concatenated_len - self._max_buffer:]
                    if self._buffer_separator is not None:
                        rest = rest.split(self._buffer_separator, 1)[-1]
                    super().seek(0)
                    super().write(rest)
                    super().truncate(super().tell() + len(__s))
            res = super().write(__s)
            if self._dup is not None:
                self._dup.write(__s)
            self._trigger(self.get_filtered_output())
            return res

        def get_filtered_output(self):
            if self._regex is None or self._buffer_separator is None:
                return self.getvalue()

            return self._buffer_separator.join(filter(self._regex.search, self.getvalue().split(self._buffer_separator)))

        def print_at_end(self):
            self._trigger(self.get_filtered_output())

    def __init__(self, stdout=None, stderr=False, format=None, to=None, max_buffer=None, buffer_separator='\\n',
                 regex=None, duplicate_out=False):
        self.io_args = {'trigger': self._write, 'max_buffer': max_buffer, 'buffer_separator': buffer_separator,
                        'regex': regex}
        self.redirections = []
        self.st = None
        self.stderr = stderr is True
        self.stdout = stdout is True or (stdout is None and not self.stderr)
        self.format = format or 'code'
        self.to = to
        self.fun = None
        self.duplicate_out = duplicate_out or None
        self.active_nested = None

        if not self.stdout and not self.stderr:
            raise ValueError("one of stdout or stderr must be True")

        if self.format not in ['text', 'markdown', 'latex', 'code', 'write']:
            raise ValueError(
                f"format need oneof the following: {', '.join(['text', 'markdown', 'latex', 'code', 'write'])}")

        if self.to and (not hasattr(self.to, 'text') or not hasattr(self.to, 'empty')):
            raise ValueError(f"'to' is not a streamlit container object")

    def __enter__(self):
        if self.st is not None:
            if self.to is None:
                if self.active_nested is None:
                    self.active_nested = self(format=self.format, max_buffer=self.io_args['max_buffer'],
                                              buffer_separator=self.io_args['buffer_separator'],
                                              regex=self.io_args['regex'], duplicate_out=self.duplicate_out)
                return self.active_nested.__enter__()
            else:
                raise Exception("Already entered")
        to = self.to or st

        to.text(f"Redirected output from "
                f"{'stdout and stderr' if self.stdout and self.stderr else 'stdout' if self.stdout else 'stderr'}"
                f"{' [' + self.io_args['regex'] + ']' if self.io_args['regex'] else ''}"
                f":")
        self.st = to.empty()
        self.fun = getattr(self.st, self.format)

        io_obj = None

        def redirect(to_duplicate):
            nonlocal io_obj
            io_obj = _Redirect.IOStuff(dup=self.duplicate_out and to_duplicate, **self.io_args)
            redirection = contextlib.redirect_stdout(io_obj)
            self.redirections.append((redirection, io_obj))
            redirection.__enter__()

        if self.stderr:
            redirect(sys.stderr)
        if self.stdout:
            redirect(sys.stdout)

        return io_obj

    def __call__(self, to=None, format=None, max_buffer=None, buffer_separator='\\n', regex=None, duplicate_out=False):
        return _Redirect(self.stdout, self.stderr, format=format, to=to, max_buffer=max_buffer,
                         buffer_separator=buffer_separator, regex=regex, duplicate_out=duplicate_out)

    def __exit__(self, *exc):
        if self.active_nested is not None:
            nested = self.active_nested
            if nested.active_nested is None:
                self.active_nested = None
            return nested.__exit__(*exc)

        res = None
        for redirection, io_obj in reversed(self.redirections):
            res = redirection.__exit__(*exc)
            io_obj.print_at_end()

        self.redirections = []
        self.st = None
        self.fun = None
        return res

    def _write(self, data):
        data = self.remove_formatting(data)
        self.fun(data)

    @staticmethod
    def remove_formatting(output):
        output = re.sub('\[[0-9;m]+', '', output)  
        output = re.sub('\', '', output)  
        output = re.sub(r'\[+', '', output)  
        output = re.sub(r'\]+', '', output)  
        output = re.sub(r'^\s*\\n', '', output)  
        output = re.sub(r'\\n+', '\\n', output)  
        output = re.sub(r'Generated (\d+) sub questions\.', r'\\nGenerated \1 sub questions.', output)  
        output = output.strip()
        return output


stdout = _Redirect()
stderr = _Redirect(stderr=True)
stdouterr = _Redirect(stdout=True, stderr=True)
"""

if st.session_state.index:
    template_filled = {
        "title" : st.session_state.title,
        "index_names" : st.session_state.file_names,
        "summaries" : st.session_state.summaries,
        "model_choice" : st.session_state.model_choice
    }
    with open("template.json", "w") as f:
        json.dump(template_filled, f)
    with open("App.py", "w") as f:
        f.write(app_code)
    f.close()
    with open("redirect.py", "w") as f:
        f.write(redirect_code)
    f.close()
    zip_file = zipfile.ZipFile("app.zip", 'w', zipfile.ZIP_DEFLATED)
    for folder in st.session_state.file_names:
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                zip_file.write(
                    os.path.join(dirpath, filename),
                    os.path.relpath(os.path.join(dirpath, filename), os.path.join(st.session_state.file_names[0], '../..')))
    zip_file.write("template.json")
    zip_file.write("App.py")
    zip_file.write("redirect.py")
    zip_file.close()
    os.remove("template.json")
    os.remove("App.py")
    os.remove("redirect.py")
    for i in st.session_state.file_names:
        for j in os.listdir(i):
            os.remove(f"{i}/{j}")
        os.rmdir(i)
    with open("app.zip", "rb") as fp:
        download = st.download_button(
            label = "Download ZIP",
            data = fp,
            file_name = "app.zip",
            mime = "application/zip",
            on_click = downloaded
        )