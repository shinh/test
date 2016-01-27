#!/usr/bin/env python

# Copyright (c) 2009, Shinichiro Hamaji
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the <ORGANIZATION> nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import difflib
import fileinput
import re
import sys

class ColorDiff:
    RED = '\033[1;31m'
    BLUE = '\033[1;34m'
    MAGENTA = '\033[1;35m'
    DARK_RED = '\033[0;31m'
    DARK_BLUE = '\033[0;34m'
    DARK_MAGENTA = '\033[0;35m'
    DARK_CYAN = '\033[0;36m'
    CANCEL = '\033[0m'

    def __init__(self):
        self.clear()

    def clear(self):
        self.minus_buf = ''
        self.plus_buf = ''

    def concatWithColor(self, dst, col, src):
        return dst + col + ''.join(src).replace('\n', '\n' + col)

    def tokenize(self, line):
        r = re.findall('[a-zA-Z0-9_]+| +|\r*\n|.', line, re.DOTALL)
        return r

    def flushAll(self):
        if self.minus_buf and self.plus_buf:
            minus = ''
            plus = ''
            minus_buf = self.tokenize(self.minus_buf)
            plus_buf = self.tokenize(self.plus_buf)
            for op, ms, me, ps, pe in difflib.SequenceMatcher(
                None, minus_buf, plus_buf).get_opcodes():
                m = minus_buf[ms:me]
                p = plus_buf[ps:pe]
                if op == 'delete':
                    minus = self.concatWithColor(minus, ColorDiff.RED, m)
                elif op == 'equal':
                    minus = self.concatWithColor(
                        minus, ColorDiff.DARK_RED, m)
                    plus = self.concatWithColor(plus, ColorDiff.DARK_BLUE, p)
                elif op == 'insert':
                    plus = self.concatWithColor(plus, ColorDiff.BLUE, p)
                elif op == 'replace':
                    minus = self.concatWithColor(minus, ColorDiff.RED, m)
                    plus = self.concatWithColor(plus, ColorDiff.BLUE, p)
            sys.stdout.write(minus + plus + ColorDiff.CANCEL)
        else:
            self.outputMinus(self.minus_buf)
            self.outputPlus(self.plus_buf)
        self.clear()

    def outputPlus(self, lines):
        sys.stdout.write(ColorDiff.BLUE + lines + ColorDiff.CANCEL)

    def outputMinus(self, lines):
        sys.stdout.write(ColorDiff.RED + lines + ColorDiff.CANCEL)

    def run(self):
        for line in fileinput.input():
            if line.startswith('-'):
                if self.plus_buf:
                    self.flushAll()
                self.minus_buf += line
            elif line.startswith('+'):
                if self.minus_buf:
                    self.plus_buf += line
                else:
                    self.outputPlus(line)
            else:
                self.flushAll()
                sys.stdout.write(line)
        self.flushAll()

ColorDiff().run()
