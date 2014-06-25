from ctypes import Structure, c_ulonglong


NULL = 0


def box_int(i):
    return i << 4 | 2;


def box_op(i):
    return i << 4 | 3;


def is_int(i):
    return i & 2;


def is_op(i):
    return i & 3;


def box(v):
    if isinstance(v, int):
        return box_int(v)
    if isinstance(v, basestring):
        return box_op(v)


def unbox(v):
    return v >> 4;


def _pair(v):
    try:
        h = box(v.next())
    except StopIteration:
        return None
    try:
        t = box(v.next())
    except StopIteration:
        return (h, NULL)
    return (h, t)


class Op(c_ulonglong):

    ops = {'+': 0,
           'iprint': 1,
           }

    iops = {v: k for k, v in ops.items()}

    def type(self):
        if is_int(self.value):
            return 'INT'
        if is_op(self.value):
            return 'OP'
        if self.value == NULL:
            return 'NULL'

    def __repr__(self):
        return '%s(%s)' % (self.type(), unbox(self.value))


class Cell(Structure):
    """
    128-bit wide packed instruction cell.

    Designed to match cuda vector width.
    """
    _fields_ = [
        ('head', Op),
        ('tail', Op),
        ]

    def __repr__(self):
        return '[%s, %s]' % (self.head, self.tail)


class Program(Structure):

    @classmethod
    def from_list(cls, vals):
        cells = []
        i = iter(vals)
        p = _pair(i)
        while p:
            cell = Cell()
            cell.head, cell.tail = p
            cells.append(cell)
            p = _pair(i)

        items = [('cells', Cell * len(cells))]
        ob = type(cls.__name__, (cls,), 
                  dict(_fields_=items))
        ob.cells = cells
        return ob

def compile(source):
    prog = []
    tokens = source.split()
    c = Cell()
    for t in token:
        try:
            i = int(t)
            prog.append(i)
            continue
        except ValueError:
            pass
        
    
