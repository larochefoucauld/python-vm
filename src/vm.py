import argparse
import builtins
import dis
import types
import typing as tp
from types import GeneratorType, CoroutineType


class ArgBinder:
    CO_VARARGS = 4
    CO_VARKEYWORDS = 8

    ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
    ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
    ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
    ERR_MISSING_POS_ARGS = 'Missing positional arguments'
    ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
    ERR_POSONLY_PASSED_AS_KW = \
        'Positional-only argument passed as keyword argument'

    UNBOUND = object()
    MISSING = object()

    def __init__(self) -> None:
        pass

    def _get_vargroups(self, func_code: types.CodeType, pos_defaults: tuple[tp.Any, ...],
                       kw_defaults: dict[str, tp.Any]) -> dict[str, list[str] | str]:
        flags = func_code.co_flags
        uses_varargs = bool(flags & self.CO_VARARGS)
        uses_varkwargs = bool(flags & self.CO_VARKEYWORDS)
        n_params = func_code.co_argcount + func_code.co_kwonlyargcount
        var_pos_name = ''
        var_kw_name = ''
        if uses_varargs and uses_varkwargs:
            var_pos_name, var_kw_name = func_code.co_varnames[n_params:n_params + 2]
        elif uses_varargs:
            var_pos_name = func_code.co_varnames[n_params]
        elif uses_varkwargs:
            var_kw_name = func_code.co_varnames[n_params]
        varnames = list(func_code.co_varnames[:n_params])
        n_posonly = func_code.co_posonlyargcount
        n_usual = func_code.co_argcount
        return {
            'all': varnames,
            'posonly': varnames[:n_posonly],
            'usual': varnames[n_posonly:n_usual],
            'kwonly': varnames[n_usual:],
            'var_pos': var_pos_name,
            'var_kw': var_kw_name,
            'pos_has_default': varnames[n_usual - len(pos_defaults):n_usual],
            'kw_has_default': list(kw_defaults.keys())}

    def _add_binding(self, bindings: dict[str, list[str] | str], name: str, val: tp.Any) -> None:
        if bindings.get(name) is self.UNBOUND:
            bindings[name] = val
        else:
            raise TypeError(self.ERR_MULT_VALUES_FOR_ARG)

    def _bind_pos_defaults(self, vargroups: dict[str, list[str] | str],
                           pos_defs: tuple[tp.Any, ...], bindings: dict[str, tp.Any]) -> None:
        default_names = vargroups['pos_has_default']
        for name, def_val in zip(default_names, pos_defs):
            if bindings[name] is self.UNBOUND:
                bindings[name] = def_val

    def _bind_kw_defaults(self, kw_defs: dict[str, tp.Any], bindings: dict[str, tp.Any]) -> None:
        for name, def_val in kw_defs.items():
            if bindings[name] is self.UNBOUND:
                bindings[name] = def_val

    def _handle_posonly_kw_match(self, vargroups: dict[str, list[str] | str],
                                 provided_kwargs: dict[str, tp.Any], bindings: dict[str, tp.Any]) -> None:
        var_kw_name = tp.cast(str, vargroups['var_kw'])
        for name in vargroups['posonly']:
            if (val := provided_kwargs.get(name, self.MISSING)) is not self.MISSING:
                if var_kw_name:
                    bindings[var_kw_name][name] = val
                    del provided_kwargs[name]
                else:
                    raise TypeError(self.ERR_POSONLY_PASSED_AS_KW)

    def _bind_posonly(self, vargroups: dict[str, list[str] | str],
                      provided_args: list[tp.Any], bindings: dict[str, tp.Any]) -> list[str]:
        posonly_names = vargroups['posonly']
        n_required = len(posonly_names)
        n_provided = len(provided_args)
        has_default = set(vargroups['pos_has_default'])
        if (n_provided < n_required and
                posonly_names[n_provided] not in has_default):
            raise TypeError(self.ERR_MISSING_POS_ARGS)
        for name, val in zip(posonly_names, provided_args):
            self._add_binding(bindings, name, val)
        return provided_args[n_required:]

    def _bind_usual(self, vargroups: dict[str, list[str] | str],
                    provided_args: list[tp.Any], provided_kwargs: dict[str, tp.Any],
                    bindings: dict[str, tp.Any]) -> None:
        usual_names = vargroups['usual']
        n_required = len(usual_names)
        n_provided = len(provided_args)
        for name, val in zip(usual_names, provided_args):
            self._add_binding(bindings, name, val)
            bindings[name] = val
        if n_provided > n_required:
            var_pos_name = tp.cast(str, vargroups['var_pos'])
            if var_pos_name:
                bindings[var_pos_name] += provided_args[n_required:]
                return
            else:
                raise TypeError(self.ERR_TOO_MANY_POS_ARGS)
        has_default = set(vargroups['pos_has_default'])
        for name in usual_names[n_provided:]:
            if (val := provided_kwargs.get(name, self.MISSING)) is not self.MISSING:
                bindings[name] = val
                del provided_kwargs[name]
            elif name not in has_default:
                raise TypeError(self.ERR_MISSING_POS_ARGS)

    def _bind_kwonly(self, vargroups: dict[str, list[str] | str],
                     unused_kwargs: dict[str, tp.Any], bindings: dict[str, tp.Any]) -> None:
        kwonly_names = vargroups['kwonly']
        has_default = set(vargroups['kw_has_default'])
        for name in kwonly_names:
            if (val := unused_kwargs.get(name, self.MISSING)) is self.MISSING:
                if name in has_default:
                    continue
                raise TypeError(self.ERR_MISSING_KWONLY_ARGS)
            self._add_binding(bindings, name, val)
            del unused_kwargs[name]
        var_kw_name = tp.cast(str, vargroups['var_kw'])
        for name, val in unused_kwargs.items():
            name_status = bindings.get(name, self.MISSING)
            if name_status is self.MISSING:
                if not var_kw_name:
                    raise TypeError(self.ERR_TOO_MANY_KW_ARGS)
                bindings[var_kw_name][name] = val
            else:
                raise TypeError(self.ERR_MULT_VALUES_FOR_ARG)

    def bind(self, func_code: types.CodeType, pos_defaults: tuple[tp.Any, ...], kw_defaults: dict[str, tp.Any],
             args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any]) -> dict[str, tp.Any]:
        vargroups = self._get_vargroups(func_code, pos_defaults, kw_defaults)
        bindings = {name: self.UNBOUND for name in vargroups['all']}
        var_pos_name = tp.cast(str, vargroups['var_pos'])
        var_kw_name = tp.cast(str, vargroups['var_kw'])
        if var_pos_name:
            bindings[var_pos_name] = []
        if var_kw_name:
            bindings[var_kw_name] = {}

        self._handle_posonly_kw_match(vargroups, kwargs, bindings)
        args_left = self._bind_posonly(vargroups, list(args), bindings)
        self._bind_usual(vargroups, args_left, kwargs, bindings)
        self._bind_kwonly(vargroups, kwargs, bindings)
        self._bind_pos_defaults(vargroups, pos_defaults, bindings)
        self._bind_kw_defaults(kw_defaults, bindings)
        if var_pos_name:
            bindings[var_pos_name] = tuple(tp.cast(list[tp.Any], bindings[var_pos_name]))
        return bindings


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    MISSING = object()  # for generic dict lookup

    OPNAMES_GEN_RETURN_ON_EXIT = {
        'RETURN_VALUE',
        'RETURN_CONST'
    }

    BINARY_OP_TABLE = {  # mapping from operation codes
        # Arithmetic
        0: lambda x, y: x + y,
        10: lambda x, y: x - y,
        5: lambda x, y: x * y,
        11: lambda x, y: x / y,
        2: lambda x, y: x // y,
        6: lambda x, y: x % y,
        4: lambda x, y: x @ y,
        8: lambda x, y: x ** y,
        # Logical and bitwise
        3: lambda x, y: x << y,
        9: lambda x, y: x >> y,
        1: lambda x, y: x & y,
        7: lambda x, y: x | y,
        12: lambda x, y: x ^ y,
    }
    BINARY_INPLACE_OP_TABLE = {  # mapping from inplace operation codes to corresponding operation codes
        # Arithmetic
        13: 0, 23: 10, 18: 5, 24: 11, 15: 2, 19: 6, 17: 4, 21: 8,
        # Logical and bitwise
        16: 3, 22: 9, 14: 1, 20: 7, 25: 12
    }

    COMPARE_OP_TABLE = {
        # Equality test
        '==': lambda x, y: x == y,
        '!=': lambda x, y: x != y,
        # Inequality test
        '<': lambda x, y: x < y,
        '<=': lambda x, y: x <= y,
        '>': lambda x, y: x > y,
        '>=': lambda x, y: x >= y
    }

    INTRINSIC_FUNC_TABLE_1 = {
        6: lambda x: list(x)
    }

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any],
                 frame_closure: tuple[tp.Any, ...]) -> None:

        self.code = frame_code
        self.instructions = list(dis.get_instructions(self.code))
        self.instr_indices = {instr.offset: i for i, instr in enumerate(self.instructions)}
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.closure = frame_closure
        self.data_stack: tp.Any = []
        self.return_value: tp.Any = None
        self.exit_required: bool = False
        self.ip = 0
        self.arg_binder = ArgBinder()
        self.call_kw_names: tuple[str, ...] | None = None

    def empty(self) -> bool:
        return not bool(self.data_stack)

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def peek(self) -> tp.Any:
        return self.MISSING if self.empty() else self.data_stack[-1]

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def _get_cur_actual_arg(self) -> tp.Any:
        return self.instructions[self.ip].arg

    def run(self) -> tp.Any:
        while self.ip < len(self.instructions) and not self.exit_required:
            # print(f"IP: {self.ip}, STACK size: {len(self.data_stack)}")  # DEBUG
            instruction = self.instructions[self.ip]
            # print(instruction)  # DEBUG
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            self.ip += 1
        return self.return_value

    def run_in_generator_mode(self) -> tp.Generator[tp.Any, tp.Any, tp.Any]:
        self.exit_required = False
        while self.ip < len(self.instructions):
            # print(f"<GEN mode> IP: {self.ip}, STACK size: {len(self.data_stack)}")  # DEBUG
            instruction = self.instructions[self.ip]
            # print(instruction)  # DEBUG
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            if self.exit_required:
                if instruction.opname in self.OPNAMES_GEN_RETURN_ON_EXIT:
                    return self.return_value
                self.exit_required = False
                got = yield self.return_value
                self.push(got)
            self.ip += 1

    def nop_op(self, arg: None) -> None:
        pass

    def pop_top_op(self, arg: tp.Any) -> None:
        self.pop()

    def end_for_op(self, arg: None) -> None:
        self.popn(2)

    def end_send_op(self, arg: None) -> None:
        self.data_stack[-2], self.data_stack[-1] = self.data_stack[-1], self.data_stack[-2]
        self.pop()

    def copy_op(self, arg: int) -> None:
        self.push(self.data_stack[-arg])

    def swap_op(self, arg: int) -> None:
        self.data_stack[-arg], self.data_stack[-1] = self.data_stack[-1], self.data_stack[-arg]

    def unary_negative_op(self, arg: None) -> None:
        self.push(-self.pop())

    def unary_not_op(self, arg: None) -> None:
        self.push(not self.pop())

    def unary_invert_op(self, arg: None) -> None:
        self.push(~self.pop())

    def get_iter_op(self, arg: None) -> None:
        self.push(iter(self.pop()))

    def get_yield_from_iter_op(self, arg: None) -> None:
        target = self.peek()
        if type(target) is GeneratorType or type(target) is CoroutineType:
            return
        self.push(iter(self.pop()))

    def binary_op_op(self, arg: int) -> None:
        y = self.pop()
        x = self.pop()
        if not (op := self.BINARY_OP_TABLE.get(arg)):
            op_code = self.BINARY_INPLACE_OP_TABLE[arg]
            op = self.BINARY_OP_TABLE[op_code]
        self.push(op(x, y))

    def binary_subscr_op(self, arg: None) -> None:
        key = self.pop()
        container = self.pop()
        self.push(container[key])

    def store_subscr_op(self, arg: None) -> None:
        key = self.pop()
        container = self.pop()
        value = self.pop()
        container[key] = value

    def delete_subscr_op(self, arg: None) -> None:
        key = self.pop()
        container = self.pop()
        del container[key]

    def binary_slice_op(self, arg: None) -> None:
        end = self.pop()
        start = self.pop()
        container = self.pop()
        self.push(container[start:end])

    def store_slice_op(self, arg: None) -> None:
        end = self.pop()
        start = self.pop()
        container = self.pop()
        values = self.pop()
        container[start:end] = values

    def set_add_op(self, arg: int) -> None:
        item = self.pop()
        set.add(self.data_stack[-arg], item)

    def list_append_op(self, arg: int) -> None:
        item = self.pop()
        list.append(self.data_stack[-arg], item)

    def map_add_op(self, arg: int) -> None:
        value = self.pop()
        key = self.pop()
        dict.__setitem__(self.data_stack[-arg], key, value)

    def return_value_op(self, arg: None) -> None:
        self.return_value = self.peek()
        self.exit_required = True

    def return_const_op(self, arg: tp.Any) -> None:
        self.return_value = arg
        self.exit_required = True

    def yield_value_op(self, arg: None) -> None:
        self.return_value = self.pop()
        self.exit_required = True

    def store_name_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def delete_name_op(self, arg: str) -> None:
        del self.locals[arg]

    def unpack_sequence_op(self, arg: int) -> None:
        self.push(*self.pop()[::-1])

    def store_attr_op(self, arg: str) -> None:
        obj = self.pop()
        value = self.pop()
        setattr(obj, arg, value)

    def delete_attr_op(self, arg: str) -> None:
        obj = self.pop()
        delattr(obj, arg)

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def load_const_op(self, arg: tp.Any) -> None:
        self.push(arg)

    def load_name_op(self, arg: str) -> None:
        for namespace in [self.locals, self.globals, self.builtins]:
            if (var := namespace.get(arg, self.MISSING)) is not self.MISSING:
                self.push(var)
                return
        raise NameError("Unknown name: " + arg)

    def build_tuple_op(self, arg: int) -> None:
        self.push(tuple(self.popn(arg)))

    def build_list_op(self, arg: int) -> None:
        self.push(list(self.popn(arg)))

    def build_set_op(self, arg: int) -> None:
        self.push(set(self.popn(arg)))

    def build_map_op(self, arg: int) -> None:
        elements = self.popn(2 * arg)
        keys = elements[::2]
        values = elements[1::2]
        self.push({key: value for key, value in zip(keys, values)})

    def build_const_key_map_op(self, arg: int) -> None:
        keys = self.pop()
        values = self.popn(arg)
        self.push({key: value for key, value in zip(keys, values)})

    def build_string_op(self, arg: int) -> None:
        strings = self.popn(arg)
        self.push(''.join(strings))

    def list_extend_op(self, arg: int) -> None:
        seq = self.pop()
        list.extend(self.data_stack[-arg], seq)

    def set_update_op(self, arg: int) -> None:
        seq = self.pop()
        set.update(self.data_stack[-arg], seq)

    def dict_update_op(self, arg: int) -> None:
        dct = self.pop()
        dict.update(self.data_stack[-arg], dct)

    def dict_merge_op(self, arg: int) -> None:
        dct = self.pop()
        target = self.data_stack[-arg]
        if set.intersection(set(target.keys()), set(dct.keys())):
            raise AttributeError
        return dict.update(target, dct)

    def load_attr_op(self, arg: str) -> None:
        actual_arg = self._get_cur_actual_arg()
        if actual_arg & 1 == 0:
            self.push(getattr(self.pop(), arg))
            return
        obj = self.pop()
        attr = getattr(obj, arg)
        if isinstance(obj, types.MethodType):
            self.push(attr.__func__, obj)
        else:
            self.push(None, attr)

    def compare_op_op(self, arg: str) -> None:
        y = self.pop()
        x = self.pop()
        op = self.COMPARE_OP_TABLE[arg]
        self.push(op(x, y))

    def is_op_op(self, arg: int) -> None:
        lhs, rhs = self.popn(2)
        if arg:
            self.push(lhs is not rhs)
        else:
            self.push(lhs is rhs)

    def contains_op_op(self, arg: int) -> None:
        lhs, rhs = self.popn(2)
        if arg:
            self.push(lhs not in rhs)
        else:
            self.push(lhs in rhs)

    def import_name_op(self, arg: str) -> None:
        level, fromlist = self.popn(2)
        module_obj = __import__(arg, fromlist=fromlist, level=level)
        self.push(module_obj)

    def import_from_op(self, arg: str) -> None:
        module_obj = self.peek()
        attr = getattr(module_obj, arg)
        self.push(attr)

    def jump_backward_op(self, arg: int) -> None:
        self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def jump_backward_no_interrupt_op(self, arg: int) -> None:
        self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def pop_jump_if_true_op(self, arg: int) -> None:
        if self.pop():
            self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def pop_jump_if_false_op(self, arg: int) -> None:
        if not self.pop():
            self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def pop_jump_if_not_none_op(self, arg: int) -> None:
        if self.pop() is None:
            self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def pop_jump_if_none_op(self, arg: int) -> None:
        if self.pop() is None:
            self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def for_iter_op(self, arg: int) -> None:
        cur_iter = self.peek()
        try:
            self.push(cur_iter.__next__())
        except StopIteration:
            next_instr_idx = self.instr_indices[arg]
            if self.instructions[next_instr_idx].opname == 'END_FOR':
                self.pop()
                self.ip = next_instr_idx  # ip increments by 1 in a loop, skip END_FOR
            else:
                self.ip = next_instr_idx - 1  # ip increments by 1 in a loop

    def load_global_op(self, arg: str) -> None:
        actual_arg = self._get_cur_actual_arg()
        push_null = actual_arg & 1
        for namespace in [self.globals, self.builtins]:
            if (var := namespace.get(arg, self.MISSING)) is not self.MISSING:
                if push_null:
                    self.push(None)
                self.push(var)
                return
        raise NameError("Unknown global: " + arg)

    def load_fast_op(self, arg: str) -> None:
        self.push(self.locals[arg])

    def load_fast_check_op(self, arg: str) -> None:
        if (val := self.locals.get(arg, self.MISSING)) is not self.MISSING:
            self.push(val)
        else:
            raise UnboundLocalError("Unbound local: " + arg)

    def load_fast_and_clear_op(self, arg: str) -> None:
        self.push(self.locals.get(arg, None))
        self.locals[arg] = None

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def delete_fast_op(self, arg: str) -> None:
        del self.locals[arg]

    def make_cell_op(self, arg: str) -> None:
        pass  # the object won't be deallocated since we pass the reference to the next frame, no need for cells

    def load_deref_op(self, arg: str) -> None:
        self.push(self.locals[arg])

    def store_deref_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def copy_free_vars_op(self, arg: int) -> None:
        free_vars = self.code.co_freevars
        bound_closure_vars = {name: val for name, val in zip(free_vars, self.closure)}
        self.locals.update(bound_closure_vars)

    def call_op(self, arg: int) -> None:
        arguments = self.popn(arg)
        if self.call_kw_names:
            n_kwargs = len(self.call_kw_names)
            args = arguments[:-n_kwargs]
            kwargs = {name: val for name, val in zip(self.call_kw_names, arguments[-n_kwargs:])}
            self.call_kw_names = None
        else:
            args = arguments
            kwargs = {}
        item1, item2 = self.popn(2)
        f, slf = None, None
        if item1 is None:  # NULL (item1) | callable (item2) | args...
            f = item2
        else:              # callable (item1) | self (item2) | args...
            f = item1
            slf = item2
        # print(f"CALL: {f=}, {slf=}, {args=}, {kwargs=}")  # DEBUG
        if slf:
            self.push(f(slf, *args, **kwargs))
        else:
            self.push(f(*args, **kwargs))

    def call_function_ex_op(self, arg: int) -> None:
        kwargs = {}
        if arg & 1:
            kwargs = self.pop()
        args = self.pop()
        f = self.pop()
        self.push(f(*args, **kwargs))

    def push_null_op(self, arg: int) -> None:
        self.push(None)

    def kw_names_op(self, arg: tuple[str, ...]) -> None:
        self.call_kw_names = arg

    def make_function_op(self, arg: int) -> None:
        code = self.pop()  # the code associated with the function (at TOS1)
        if arg & 8:  # closure variables
            closure_vars = self.pop()
        else:
            closure_vars = ()
        if arg & 4:  # param annotations
            self.pop()  # ignore for now
        kw_defaults = self.pop() if arg & 2 else {}
        pos_defaults = self.pop() if arg & 1 else ()

        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            arg_bindings = self.arg_binder.bind(code, pos_defaults, kw_defaults, args, kwargs)
            func_locals = dict(self.locals)
            func_locals.update(arg_bindings)
            frame = Frame(code, self.builtins, self.globals, func_locals, closure_vars)
            return frame.run()

        self.push(wrapper)

    def build_slice_op(self, arg: int) -> None:
        if arg == 2:
            end = self.pop()
            start = self.pop()
            self.push(slice(start, end))
        elif arg == 3:
            step = self.pop()
            end = self.pop()
            start = self.pop()
            self.push(slice(start, end, step))

    def format_value_op(self, arg: tuple[tp.Any, bool]) -> None:
        trf, fmt = arg
        fmt_spec = self.pop() if fmt else ''
        value = self.pop()
        self.push(format(trf(value), fmt_spec))

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def return_generator_op(self, arg: None) -> None:
        self.push(None)  # to handle following POP_TOP
        self.return_value = self.run_in_generator_mode()
        self.exit_required = True

    def send_op(self, arg: int) -> None:
        val = self.pop()
        target = self.peek()
        try:
            if val is None and hasattr(target, '__next__'):
                self.push(next(target))
            else:
                self.push(target.send(val))
        except StopIteration as exc:
            self.push(exc.value)
            self.ip = self.instr_indices[arg] - 1  # ip increments by 1 in a loop

    def call_intrinsic_1_op(self, arg: int) -> None:
        argument = self.pop()
        result = self.INTRINSIC_FUNC_TABLE_1[arg](argument)
        self.push(result)

    def load_closure_op(self, arg: str) -> None:
        self.push(self.locals[arg])


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.__dict__, globals_context, globals_context, ())
        return frame.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM arguments")
    parser.add_argument("file", type=str, help="Source file")
    args = parser.parse_args()

    with open(args.file) as f:
        code = compile(f.read(), args.file, 'exec')
        VirtualMachine().run(code)
