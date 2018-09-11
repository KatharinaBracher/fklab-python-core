import io, os, sys, types

import ast

from nbformat import read

#from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

__all__ = ['install_notebook_loader']

# code borrowed from:
# https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Importing%20Notebooks.html#

# with additional code filtering idea from:
# https://github.com/ipython/ipynb

def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path

    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    name = fullname.rsplit('.', 1)[-1]
    if not path:
        path = ['']
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path


ALLOWED_NODES = set([
    ast.ClassDef,
    ast.FunctionDef,
    ast.Import,
    ast.ImportFrom
])

def filter_ast(module_ast):
    """
    Filters a given module ast, removing non-whitelisted nodes
    It allows only the following top level items:
     - imports
     - function definitions
     - class definitions
     - top level assignments where all the targets on the LHS are all caps
    """
    def node_predicate(node):
        """
        Return true if given node is whitelisted
        """
        for an in ALLOWED_NODES:
            if isinstance(node, an):
                return True

        # Recurse through Assign node LHS targets when an id is not specified,
        # otherwise check that the id is uppercase
        if isinstance(node, ast.Assign):
            return all([node_predicate(t) for t in node.targets if not hasattr(t, 'id')]) \
                and all([t.id.isupper() for t in node.targets if hasattr(t, 'id')])

        return False

    module_ast.body = [n for n in module_ast.body if node_predicate(n)]
    return module_ast

class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""
    def __init__(self, path=None, defs_only=False):
        self.shell = InteractiveShell.instance()
        self.path = path
        self.defs_only = defs_only
    
    def code_from_ipynb(self, nb, markdown=False):
        """
        Get the code for a given notebook
        nb is passed in as a dictionary that's a parsed ipynb file
        """
        code = ""
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # transform the input to executable Python
                code += ''.join(self.shell.input_transformer_manager.transform_cell(cell.source))
                #code += ''.join(cell.source)
            if markdown and cell.cell_type == 'markdown':
                code += '\n# ' + '# '.join(cell.source)
            # We want a blank newline after each cell's output.
            # And the last line of source doesn't have a newline usually.
            code += '\n\n'
        return code

    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.path)

        #print ("importing Jupyter notebook from %s" % path)

        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = read(f, 4)


        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        #mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod

        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__

        try:
            code = ast.parse(self.code_from_ipynb(nb))
            if self.defs_only:
                code = filter_ast(code)
            exec(compile(code, filename="<ast>", mode="exec"), mod.__dict__)               
        finally:
            self.shell.user_ns = save_user_ns
            
        return mod

class NotebookFinder(object):
    """Module finder that locates Jupyter Notebooks"""
    def __init__(self, *args, **kwargs):
        self.loaders = {}
        self.args, self.kwargs = args, kwargs

    def find_module(self, fullname, path=None):
        nb_path = find_notebook(fullname, path)
        if not nb_path:
            return

        key = path
        if path:
            # lists aren't hashable
            key = os.path.sep.join(path)

        if key not in self.loaders:
            self.loaders[key] = NotebookLoader(path, *self.args, **self.kwargs)
        return self.loaders[key]

def install_notebook_loader(defs_only=True):
    sys.meta_path = [x for x in sys.meta_path if not isinstance(x,NotebookFinder)]
    sys.meta_path.append(NotebookFinder(defs_only=defs_only))
