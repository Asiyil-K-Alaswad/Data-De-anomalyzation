import urllib.parse, json

AREAS = [110,111,120,121,130,131,141,142,143,150,151,160,161,171,181,
         1130,1151,1160,1171,1110,14101,15101,10101,1]
q = ("SELECT net, area, count() n, round(avg(lon),6) mx, round(avg(lat),6) my, "
     "round(avg(lon*lon)-avg(lon)*avg(lon),9) vxx, "
     "round(avg(lat*lat)-avg(lat)*avg(lat),9) vyy, "
     "round(avg(lon*lat)-avg(lon)*avg(lat),9) vxy, "
     "round(min(lon),4) w, round(max(lon),4) e, round(min(lat),4) s, round(max(lat),4) nn "
     "FROM cell_towers WHERE mcc=427 AND area IN (%s) "
     "GROUP BY net, area ORDER BY net, area FORMAT JSON" % ",".join(map(str, AREAS)))
url = "https://play.clickhouse.com/?user=play&query=" + urllib.parse.quote(q)
print(url)
