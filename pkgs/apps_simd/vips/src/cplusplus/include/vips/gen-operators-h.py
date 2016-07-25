#!/usr/bin/python

# walk vips and generate headers for all operators

# sample member declaration :

# VImage invert( VOption *options = 0 )

import sys
import re

import logging
#logging.basicConfig(level = logging.DEBUG)

from gi.repository import Vips, GObject

vips_type_image = GObject.GType.from_name("VipsImage")
vips_type_operation = GObject.GType.from_name("VipsOperation")
param_enum = GObject.GType.from_name("GParamEnum")

# turn a GType into a C++ type
gtype_to_cpp = {
    "VipsImage" : "VImage",
    "gint" : "int",
    "gdouble" : "double",
    "gboolean" : "bool",
    "gchararray" : "char *",
    "VipsArrayDouble" : "std::vector<double>",
    "VipsArrayImage" : "std::vector<VImage>",
    "VipsBlob" : "VipsBlob *"
}

def get_ctype(prop):
    # enum params use the C name as their name
    if GObject.type_is_a(param_enum, prop):
        return prop.value_type.name

    return gtype_to_cpp[prop.value_type.name]

def find_required(op):
    required = []
    for prop in op.props:
        flags = op.get_argument_flags(prop.name)
        if not flags & Vips.ArgumentFlags.REQUIRED:
            continue
        if flags & Vips.ArgumentFlags.DEPRECATED:
            continue

        required.append(prop)

    def priority_sort(a, b):
        pa = op.get_argument_priority(a.name)
        pb = op.get_argument_priority(b.name)

        return pa - pb

    required.sort(priority_sort)

    return required

# find the first input image ... this will be used as "this"
def find_first_input_image(op, required):
    found = False
    for prop in required:
        flags = op.get_argument_flags(prop.name)
        if not flags & Vips.ArgumentFlags.INPUT:
            continue
        if GObject.type_is_a(vips_type_image, prop.value_type):
            found = True
            break

    if not found:
        return None

    return prop

# find the first output arg ... this will be used as the result
def find_first_output(op, required):
    found = False
    for prop in required:
        flags = op.get_argument_flags(prop.name)
        if not flags & Vips.ArgumentFlags.OUTPUT:
            continue
        found = True
        break

    if not found:
        return None

    return prop

# swap any "-" for "_"
def cppize(name):
    return re.sub('-', '_', name)

def gen_arg_list(op, required):
    first = True
    for prop in required:
        if not first:
            print ',',
        else:
            first = False

        print get_ctype(prop),

        # output params are passed by reference
        flags = op.get_argument_flags(prop.name)
        if flags & Vips.ArgumentFlags.OUTPUT:
            print '*',

        print cppize(prop.name),

    if not first:
        print ',',
    print 'VOption *options = 0',

def gen_operation(cls):
    op = Vips.Operation.new(cls.name)
    gtype = Vips.type_find("VipsOperation", cls.name)
    nickname = Vips.nickname_find(gtype)
    all_required = find_required(op)

    result = find_first_output(op, all_required)
    this = find_first_input_image(op, all_required)

    # shallow copy
    required = all_required[:]
    if result != None:
        required.remove(result)
    if this != None:
        required.remove(this)

    # no "this" available, it's a class method
    if this == None:
        print 'static',

    if result == None:
        print 'void',
    else:
        print '%s' % gtype_to_cpp[result.value_type.name],

    print '%s(' % nickname,

    gen_arg_list(op, required)

    print ');'

# we have a few synonyms ... don't generate twice
generated = {}

def find_class_methods(cls):
    if not cls.is_abstract():
        gtype = Vips.type_find("VipsOperation", cls.name)
        nickname = Vips.nickname_find(gtype)
        if not nickname in generated:
            gen_operation(cls)
            generated[nickname] = True

    if len(cls.children) > 0:
        for child in cls.children:
            find_class_methods(child)

if __name__ == '__main__':
    find_class_methods(vips_type_operation)

