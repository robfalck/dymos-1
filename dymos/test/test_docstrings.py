import os
import unittest

from numpydoc.validate import validate

import dymos
from dymos.test.test_pep8 import _discover_python_files


IGNORE = ['SA01', 'EX01', 'ES01', 'SS03']


class TestDocstrings(unittest.TestCase):

    def test_docstrings(self):
        """ Tests that all files in this, directory, the parent directory, and test
        sub-directories are PEP8 compliant.

        Notes
        -----
        max_line_length has been set to 130 for this test.
        """
        import ast


        dymos_path = os.path.split(dymos.__file__)[0]
        pyfiles = _discover_python_files(dymos_path)

        for filename in pyfiles:
            with open(filename) as file:
                node = ast.parse(file.read())
                functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
                if functions:
                    print(functions)
                if classes:
                    print(classes)


        #
        # for importer, modname, ispkg in pkgutil.walk_packages(path=dm.__path__,
        #                                                       prefix=dm.__name__ + '.',
        #                                                       onerror=lambda x: None):
        #     if ispkg:
        #         continue
        #     if '.doc.' in modname or '.test.' in modname:
        #         continue
        #
        #     print(modname)
        #
        #     for name, obj in inspect.getmembers(modname, inspect.isclass):
        #         print(name)
        #         # path = repr(obj)
        #         # obj_path = path[path.index("'")+1:path.rindex("'")]
        #         # print(obj_path)
        #
        # #     for name, obj in inspect.getmembers('dymos', inspect.isclass):
        # #         path = repr(obj)
        # #         obj_path = path[path.index("'")+1:path.rindex("'")]
        # #         errors = []
        # #         for e in validate(obj_path)['errors']:
        # #             if e[0] in IGNORE:
        # #                 continue
        # #             if e[1] == "Parameters {'**kwargs'} not documented":
        # #                 continue
        # #             errors.append(e)
        # #
        # #         if not errors:
        # #             continue
        # #
        # #         print(obj_path)
        # #         for e in errors:
        # #             print('    ', e)
        # #
        # #         # print(name, obj, repr(obj), obj.__module__, obj.__class__)
        # #         # print(dir(obj))
        # #     # print(modname)
        # #     # for e in validate(modname)['errors']:
        # #     #     print(e)
        # #
        # # # if report.total_errors > 0:
        # # #     self.fail("Found pep8 errors:\n%s" % msg.getvalue())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
