import sys
from pathlib import Path

import numpy as np


def data_field_property(field):
    if isinstance(field, int):
        return property(lambda self: self._data[self.FIELD[field]])
    else:
        return property(lambda self: self._data[field])


def delegate_property(attr: str, prop):
    if isinstance(prop, str):
        return property(lambda self: getattr(getattr(self, attr), prop))
    else:
        return property(lambda self: prop.fget(getattr(self, attr)))


def abstract_property(message="abstract property"):
    def _raise(self):
        raise RuntimeError(message)

    return property(_raise)


class DataBase:
    __slots__ = ("_data",)

    def __init__(self, data):
        # unwrap data
        while isinstance(data, DataBase):
            data = data._data

        self._data: np.ndarray = data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def fields(self):
        return self._data.dtype.names

    @property
    def fmt(self):
        d = self._data.dtype

        def _fmt(t):
            if t == np.int:
                return "%d"
            elif t == np.float:
                return "%f"
            else:
                return "%s"

        return [_fmt(d[i]) for i in range(len(d))]

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        s = self._data.shape
        if len(s) == 0:
            return 1
        return s[0]

    def __getitem__(self, item):
        return type(self)(self._data[item])

    def __setitem__(self, item, value):
        self._data[item] = value

    def __delitem__(self, item):
        del self._data[item]

    def __iter__(self):
        for i in range(len(self)):
            yield type(self)(self._data[i])

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    def print_table(self):
        f = self.fields
        if f is not None:
            t = Table(*f)
            for d in self._data:
                t.append(*d)
            t.print()

    def extend(self, *fields: str, dtype=np.float):
        r, _ = self._data.shape
        self._data = np.hstack((self._data, np.zeros((r, len(fields)), dtype=dtype)))

    # load/save

    def copy(self):
        return type(self)(self._data.copy())

    @classmethod
    def zeros(cls, shape):
        dtype = np.dtype(list(zip(cls.FIELD, cls.DTYPE)))
        return cls(np.zeros(shape, dtype=dtype))

    @classmethod
    def new(cls, *a):
        dtype = np.dtype(list(zip(cls.FIELD, cls.DTYPE)))

        return cls(np.array(list(zip(*a)), dtype=dtype))

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save(self, file, file_type=None):
        if file_type is None:
            if isinstance(file, str):
                _, file_type = part_suffix(file, ".", from_right=True)
            elif isinstance(file, Path):
                _, file_type = part_suffix(file.name, ".", from_right=True)
            else:
                file_type = "csv"

        if file_type not in ("npy", "csv", "tsv"):
            raise ValueError(f"unsupported file type : {file_type}")

        if file is None:
            file = sys.stdout

        if isinstance(file, str):
            with open(file, "w") as f:
                self.save(f, file_type=file_type)

        elif isinstance(file, Path):
            with file.open("w") as f:
                self.save(f, file_type=file_type)

        else:
            if file_type == "npy":
                np_save(file, self._data)

            elif file_type == "csv":
                np_savetxt(
                    file,
                    self._data,
                    fmt=self.fmt,
                    delimiter=",",
                    comments="",
                    header=",".join(self.fields),
                )

            elif file_type == "tsv":
                np_savetxt(
                    file,
                    self._data,
                    fmt=self.fmt,
                    delimiter="\t",
                    comments="",
                    header="\t".join(self.fields),
                )

            else:
                raise ValueError(f"unsupported file type : {file_type}")

    # numpy extend

    def split(self, index):
        return [type(self)(d) for d in np.split(self._data, index)]


class FieldExpr:
    _unwrap_warn = set()

    __slots__ = "_name", "_formatter", "_filter"

    def __init__(self, name):
        self._name = name
        self._formatter = None
        self._filter = None

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return self._name

    def __repr__(self):
        return "FieldExpr:" + self._name

    def get_attr(self, o, index=None):
        a = o[self._name]

        if index is not None:
            a = a[index]

        return a

    def filter(self, o):
        return np.apply_over_axes(lambda a, x: self.test(a), o._data[self._name], 0)

    def repr(self, value):
        if self._formatter is None:
            return repr(value)
        else:
            return self._formatter % value

    def test(self, value):
        if self._filter is None:
            return np.ones(len(value), dtype=bool)
        else:
            return self._filter(value)

    @classmethod
    def parse(cls, expr: str, *, parse_format=True, parse_filter=True):
        if "=" in expr and parse_filter:
            n, e = part_suffix(expr, "=")
        else:
            n = expr
            e = None

        if "%" in n and parse_format:
            n, f = part_suffix(n, "%")
            f = "%" + f
        else:
            f = None

        if e is not None:
            e = cls._parse_expr(e)
            e = eval(f"lambda it, np=np: {e}", {}, {"np": np})

        ret = FieldExpr(n)
        ret._formatter = f
        ret._filter = e

        return ret

    @classmethod
    def _parse_expr(cls, expr: str):
        if "||" in expr:
            p = ",".join(map(lambda e: cls._parse_expr(e), expr.strip().split("||")))
            return f"np.logical_or.reduce(({p}))"
        elif "&&" in expr:
            p = ",".join(map(lambda e: cls._parse_expr(e), expr.strip().split("&&")))
            return f"np.logical_and.reduce(({p}))"
        else:
            return expr.strip()


def part_suffix(line: str, char: str, missing=None, from_right=False):
    try:
        if from_right:
            i = line.rindex(char)
        else:
            i = line.index(char)
        return line[:i], line[i + len(char) :]
    except ValueError:
        return line, missing


def _line_max_length(ls) -> int:
    length = 0

    for s in ls:
        length = max(length, len(s))

    return length


def list_padding(ls, ele=None, *, align_right=False, padding=" ", split=" "):
    if ele is None:
        length = _line_max_length(ls)

        for i, ln in enumerate(ls):
            ls[i] = ln + split + padding * (length - len(ln))

    else:
        if len(ls) != len(ele):
            raise RuntimeError("size not match %d != %d" % (len(ls), len(ele)))

        if align_right:
            length = _line_max_length(ls)
            ele_length = _line_max_length(ele)

            for i in range(len(ls)):
                ls[i] = (
                    ls[i]
                    + padding * (length - len(ls[i]))
                    + split
                    + padding * (ele_length - len(ele[i]))
                    + ele[i]
                )

        else:
            length = _line_max_length(ls)

            for i in range(len(ls)):
                ls[i] = ls[i] + padding * (length - len(ls[i])) + split + ele[i]


class Table:
    def __init__(self, *title: str, column: int = None, align_right=True):
        if len(title) == 0:
            if column is None:
                raise ValueError("lost column")
            self._cols = [[] for _ in range(column)]
        else:
            self._cols = [[t] for t in title]

        self._fmt = [{"align_right": align_right} for _ in self._cols]
        self.float_format = "%f"

    @property
    def columns(self) -> int:
        return len(self._cols)

    @property
    def rows(self) -> int:
        return len(self._cols[0])

    def get_column(self, text: str):
        for c in range(len(self._cols)):
            if self._cols[c][0] == text:
                return c
        return None

    def _str(self, item) -> str:
        if isinstance(item, float):
            return self.float_format % item
        else:
            return str(item)

    def append(self, *content):
        for i, col in enumerate(self._cols):
            try:
                elem = content[i]
            except IndexError:
                col.append("")
            else:
                col.append(self._str(elem) if elem is not None else "")

    def extend(self, content):
        for line in content:
            for i, col in enumerate(self._cols):
                try:
                    elem = line[i]
                except IndexError:
                    col.append("")
                else:
                    col.append(self._str(elem) if elem is not None else "")

    def set_format(self, column, align_right=True, padding=" ", split=None):
        f = {"align_right": align_right, "padding": padding}

        if split is not None:
            f["split"] = split

        if column is None:
            for c in range(self.columns):
                self._fmt[c] = f

        elif isinstance(column, int):
            self._fmt[column] = f

        elif isinstance(column, str):
            c = self.get_column(column)
            if c is not None:
                self._fmt[c] = f
        elif isinstance(column, (tuple, list)):
            if len(column):
                if isinstance(column[0], int):
                    for c in column:
                        self._fmt[c] = f
                else:
                    for t in column:
                        c = self.get_column(t)
                        if c is not None:
                            self._fmt[c] = f
        else:
            raise TypeError("column")

    def lines(self):
        if self._fmt[0]["align_right"] is False:
            ret = list(self._cols[0])
            skip = 1
        else:
            ret = ["" for _ in self._cols[0]]
            skip = 0

        for i, col in enumerate(self._cols):
            if i >= skip:
                list_padding(ret, col, **self._fmt[i])

        return ret

    def print(self, indent="", file=None):
        if isinstance(indent, int):
            indent = " " * (indent - 1)

        for line in self.lines():
            print(indent, line, sep="", file=file)

    def dump(self, file=None):
        for r in range(len(self._cols[0])):
            print(",".join(map(lambda col: col[r], self._cols)), file=file)
