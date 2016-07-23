#!/usr/bin/python

import json
import sys

from PIL import Image

def name_filter(c):
    if c > 200:
        return 255
    else:
        return 0

def image_diff(i0, i1):
    w, h = i0.size
    d = 0
    for y in xrange(h):
        for x in xrange(w):
            r0, g0, b0 = i0.getpixel((x, y))
            r1, g1, b1 = i1.getpixel((x, y))
            d += abs(r0-r1) + abs(g0-g1) + abs(b0-b1)
    return d / 255

def is_same(i0, i1):
    if image_diff(i0, i1) < 500:
        return True
    else:
        return False

def kill_rate(k, d):
    if d == 0:
        d = 1
    return '%.2f' % (float(k) / d)

Y_BASES = []
for i in xrange(8):
    if i < 4:
        y = 108
    else:
        y = 438
    y += i % 4 * 65
    Y_BASES.append(y)

imgs = sys.argv[2:]

records = []
with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line)
        time = int(data['time'])
        # TODO: do something better.
        #if time > 1462633200 and time < 1462719600:
        #if time > 1462798800 and time < 1462806000:
        #if time > 1465225200:
        if 'lobby' not in data or data['lobby'] != 'private':
            continue
        if 'players' not in data:
            continue
        if time > 1468335600:
            records.append(data)

while len(records) < len(imgs):
    imgs.pop(0)

name_imgs = []

assert len(imgs) == len(records)

for gi, a in enumerate(imgs):
    img = Image.open(a)
    max_index = -1
    max_color = 0
    for i, y in enumerate(Y_BASES):
        r, g, b = img.getpixel((620, y+3))
        c = r + g + b
        if max_color < c:
            max_color = c
            max_index = i

    for i, y in enumerate(Y_BASES):
        x = 808
        if i == max_index:
            x -= 41
        ni = img.crop((x, y, x + 192, y + 24))
        w, h = ni.size
        for y in xrange(h):
            for x in xrange(w):
                r, g, b = ni.getpixel((x, y))
                c = r + g + b
                if c > 600:
                    c = 255
                else:
                    c = 0
                ni.putpixel((x, y), (c, c, c))

        nid = -1
        for j, pni in enumerate(name_imgs):
            if is_same(ni, pni):
                nid = j
                break
        else:
            nid = len(name_imgs)
            ni.save('name_imgs/%03d.png' % nid)
            name_imgs.append(ni)
        records[gi]['players'][i]['img'] = nid

table = []
row = [''] + ['<img src="name_imgs/%03d.png">' % i
              for i in xrange(len(name_imgs))]
table.append(row)

cnts = [[0, 0, 0] for _ in name_imgs]
for gi, record in enumerate(records):
    team_id = int(record['team'])
    has_won = record['result'] == 'win'

    row = ['%s / %s' % (record['map'], record['rule'])]
    row += [''] * len(name_imgs)
    for player in record['players']:
        nid = player['img']
        kills = int(player['kills'])
        deaths = int(player['deaths'])
        color = ['#fcc', '#cfc'][(team_id == int(player['team'])) == has_won]
        row[nid+1] = (
            '<span style="background-color: %s">%sk%sd %s' %
            (color, kills, deaths, kill_rate(kills, deaths)))
        cnts[nid][0] += 1
        cnts[nid][1] += kills
        cnts[nid][2] += deaths
    table.append(row)

row = ['total']
for cnt, kills, deaths in cnts:
    row.append('%sk%sd %s' % (kills, deaths, kill_rate(kills, deaths)))
table.append(row)

print '<table border=1>'
for y, row in enumerate(table):
    print '<tr>'
    for x, col in enumerate(row):
        tag = 'td'
        if x == 0 or y == 0:
            tag = 'th'
        print '<%s>' % tag
        print col
        print '</%s>' % tag
    print '</tr>'
print '</table>'
