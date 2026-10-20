import urllib.parse
AREAS=[110,111,120,121,130,131,141,142,143,150,151,160,161,171,181,
       1110,1130,1151,1160,1171,10101,14101,15101]
sel=("SELECT area, count() n, "
     "round(quantileExact(0.1)(lon),3) x10, round(quantileExact(0.25)(lon),3) x25, "
     "round(quantileExact(0.5)(lon),3) x50, round(quantileExact(0.75)(lon),3) x75, "
     "round(quantileExact(0.9)(lon),3) x90, "
     "round(quantileExact(0.1)(lat),3) y10, round(quantileExact(0.25)(lat),3) y25, "
     "round(quantileExact(0.5)(lat),3) y50, round(quantileExact(0.75)(lat),3) y75, "
     "round(quantileExact(0.9)(lat),3) y90 "
     "FROM cell_towers WHERE mcc=427 AND net=2 AND radio %s AND area IN (%s) "
     "GROUP BY area ORDER BY area FORMAT JSON")
for tag,cond in (("GSM","= 'GSM'"),("UTL","IN ('UMTS','LTE')")):
    q=sel % (cond, ",".join(map(str,AREAS)))
    print(tag, "https://play.clickhouse.com/?user=play&query="+urllib.parse.quote(q))
