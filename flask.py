import sqlite3
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import base64
from io import BytesIO
from datetime import datetime as dt
import numpy as np
import streamlit as st
import os

st.set_page_config(layout="wide")

mymin = st.query_params.get_all('min')
if len(mymin)==0:
    mymin.append(300)
mymin = mymin[0]
mymax = st.query_params.get_all('max')
if len(mymax)==0:
    mymax.append(1000)
mymax = mymax[0]
mytype = st.query_params.get_all('type')
if len(mytype)==0:
    mytype.append('turns')
mytype = mytype[0]
mydb = st.query_params.get_all('db')
if len(mydb)==0:
    mydb.append('bga2')
mydb = mydb[0]
mymap = st.query_params.get_all('map')
if len(mymap)==0:
    mymap.append('all')
mymap = mymap[0]

def join_files(input_files, output_file):
    with open(output_file, "wb") as outfile:
        for filename in input_files:
            with open(filename, "rb") as infile:
                outfile.write(infile.read())

flist = [x for x in os.listdir('.') if x.find('bga.db') >= 0]
if len(flist) == 0:
    flist = [x for x in os.listdir('.') if x.find('bgadb') >= 0]
    flist.sort()
    join_files(flist, 'bga.db')

st.write(str(mymin) + str(mymax) + str(mytype) + str(mydb) + str(mymap))
start = dt.now()
if mydb == 'bga1':
    connection = sqlite3.connect('bga.db')
    sq1 = f'SELECT "table" FROM games GROUP BY "table" HAVING MAX(CAST(elo AS INT)) >= {mymin} AND MAX(CAST(elo AS INT)) <= {mymax}'
    sq2 = 'SELECT "table" FROM arknovap GROUP BY "table" HAVING COUNT(*) = 2'
    df3a = pd.read_sql(f'SELECT "Number of turns" as turns, Score as Score2, Map FROM arknovap WHERE (("Game result" LIKE "%1st%" AND CAST(Score as INT) >= 100) OR "Triggered end of game" = "Yes") AND "table" IN ({sq1}) AND "table" IN ({sq2})',connection)
    connection.close()
    df3a['turns'] = df3a['turns'].astype('int')
    df3a['Score2'] = df3a['Score2'].astype('int')    
    df3a['perturn'] = df3a['Score2'] / df3a['turns']
    if mytype != 'perturn':
        df7a=df3a.groupby(['Map']).agg({'turns':['count','median','mean',('10 pct', lambda x: x.quantile(0.1)),('90 pct', lambda x: x.quantile(0.9))]})
        df7a.columns = [' '.join(col).strip() for col in df7a.columns.values]
        df7a.loc['All Maps'] = [df3a['turns'].describe()['count'],df3a['turns'].describe()['50%'],df3a['turns'].describe()['mean'],df3a['turns'].quantile(0.1),df3a['turns'].quantile(0.9)]
        df7a = df7a.sort_values(['turns 10 pct']).reset_index()
    else:
        df6a=df3a.groupby(['Map']).agg({'perturn':['count','median','mean',('10 pct', lambda x: x.quantile(0.1)),('90 pct', lambda x: x.quantile(0.9))]})
        df6a.columns = [' '.join(col).strip() for col in df6a.columns.values]
        df6a.loc['All Maps'] = [df3a['perturn'].describe()['count'],df3a['perturn'].describe()['50%'],df3a['perturn'].describe()['mean'],df3a['perturn'].quantile(0.1),df3a['perturn'].quantile(0.9)]
        df6a = df6a.sort_values(['perturn 10 pct'],ascending=False).reset_index()
else:
    connection = sqlite3.connect('bga2flask.db')
    df = pd.read_sql(f'SELECT MAP,turns,perturn,SUM(count) as count FROM arknovac WHERE elo >= {mymin} AND elo <= {mymax} GROUP BY MAP,turns,perturn',connection)
    connection.close()
    #connection = sqlite3.connect('bga.db')
    #sq1 = f'SELECT "table" FROM games GROUP BY "table" HAVING MAX(CAST(elo2 AS INT)) >= {mymin} AND MAX(CAST(elo2 AS INT)) <= {mymax}'
    #sq2 = 'SELECT "table" FROM arknovap GROUP BY "table" HAVING COUNT(*) = 2'
    #df3a = pd.read_sql(f'SELECT "Number of turns2" as turns, Score2, Map FROM arknovap WHERE (("Game result" LIKE "%1st%" AND Score2 >= 100) OR "Triggered end of game" = "Yes") AND "table" IN ({sq1}) AND "table" IN ({sq2})',connection)
    #connection.close()
    #df3a['perturn'] = df3a['Score2'] / df3a['turns']
    #df = df3a
    #df['count']=1
    if mytype != 'perturn':
        maplist = []
        for i in df['Map'].drop_duplicates().tolist():
            df2 = df[df['Map'] == i].sort_values(['turns'])
            df2['sp'] = df2['turns'] * df2['count']
            df2['cs'] = df2['count'].cumsum()
            mycount = df2['count'].sum()
            mymean = df2['sp'].sum() / mycount
            myp10 = df2[df2['cs'] >= mycount * 0.1]['turns'].min()
            mymedian = df2[df2['cs'] >= mycount * 0.5]['turns'].min()
            myp90 = df2[df2['cs'] >= mycount * 0.9]['turns'].min()
            maplist.append([i,mycount,mymean,myp10,mymedian,myp90])
        df2 = df.sort_values(['turns'])
        df2['sp'] = df2['turns'] * df2['count']
        df2['cs'] = df2['count'].cumsum()
        mycount = df2['count'].sum()
        mymean = df2['sp'].sum() / mycount
        myp10 = df2[df2['cs'] >= mycount * 0.1]['turns'].min()
        mymedian = df2[df2['cs'] >= mycount * 0.5]['turns'].min()
        myp90 = df2[df2['cs'] >= mycount * 0.9]['turns'].min()
        maplist.append(['All Maps',mycount,mymean,myp10,mymedian,myp90])
        df7a = pd.DataFrame(maplist,columns=['Map','turns count','turns mean','turns 10 pct','turns median','turns 90 pct'])
        df7a = df7a.sort_values(['turns 10 pct'])
    else:
        maplist = []
        for i in df['Map'].drop_duplicates().tolist():
            df2 = df[df['Map'] == i].sort_values(['perturn'],ascending=False)
            df2['sp'] = df2['perturn'] * df2['count']
            df2['cs'] = df2['count'].cumsum()
            mycount = df2['count'].sum()
            mymean = df2['sp'].sum() / mycount
            myp10 = df2[df2['cs'] >= mycount * 0.1]['perturn'].max()
            mymedian = df2[df2['cs'] >= mycount * 0.5]['perturn'].max()
            myp90 = df2[df2['cs'] >= mycount * 0.9]['perturn'].max()
            maplist.append([i,mycount,mymean,myp10,mymedian,myp90])
        df2 = df.sort_values(['perturn'],ascending=False)
        df2['sp'] = df2['perturn'] * df2['count']
        df2['cs'] = df2['count'].cumsum()
        mycount = df2['count'].sum()
        mymean = df2['sp'].sum() / mycount
        myp10 = df2[df2['cs'] >= mycount * 0.1]['perturn'].max()
        mymedian = df2[df2['cs'] >= mycount * 0.5]['perturn'].max()
        myp90 = df2[df2['cs'] >= mycount * 0.9]['perturn'].max()
        maplist.append(['All Maps',mycount,mymean,myp10,mymedian,myp90])
        df6a = pd.DataFrame(maplist,columns=['Map','perturn count','perturn mean','perturn 10 pct','perturn median','perturn 90 pct'])
        df6a = df6a.sort_values(['perturn 10 pct'],ascending=False)

#    df = pd.read_sql(f'SELECT * FROM arknovap',connection)
#    df2 = df.groupby(['table']).agg({'Number of turns':'count'}).reset_index()
#    df2list = df2[df2['Number of turns']==2]['table'].tolist()
#    df3 = df[((df['Game result'].str.contains('1st')) & (df['Game result'].str.len() >=9)) | (df['Triggered end of game'] == 'Yes')]
#    df3 = df3[df3['table'].isin(df2list)]
#    df4 = pd.read_sql(f'SELECT * FROM games',connection)
#    df4['elo']=df4['elo'].str.replace('mer','0').astype('float')
#    df5 = df4.groupby(['table']).agg({'elo':'max'}).reset_index()
#    df3['Number of turns'] = df3['Number of turns'].astype('int')
#    df3 = df3.rename(columns={'Number of turns':'turns'})
#    df3['perturn'] = df3['Game result'].str[-4:-1]
#    df3['perturn'] = df3['perturn'].str.replace('(','').astype('int') / df3['turns']
#    df5list = df5[(df5['elo'] >= mymin) & (df5['elo'] <= mymax)]['table'].tolist()
#    df3a = df3[df3['table'].isin(df5list)]
fig = Figure(figsize=(18, 12))
ax = fig.subplots()
if mytype != 'perturn':
    if mymap == 'all':
        df7a.plot.scatter(y='Map',x='turns median',ax=ax,color='b',s=100,label=f'Elo {mymin}-{mymax}')
        df7a.plot.scatter(y='Map',x='turns 10 pct',ax=ax,color='b',marker='>',s=100)
        df7a.plot.scatter(y='Map',x='turns 90 pct',ax=ax,color='b',marker='<',s=100)
        df7a.plot.scatter(y='Map',x='turns mean',ax=ax,color='b',marker='^',s=100)
        for i in range(len(df7a)):
            line1 = Line2D([df7a.iloc[i]['turns 10 pct'],df7a.iloc[i]['turns 90 pct']],[df7a.iloc[i]['Map'],df7a.iloc[i]['Map']],color='b',linewidth=10,alpha=0.2)
            ax.add_line(line1)
#            ax.annotate(int(df7a.iloc[i]['turns count']),(df7a.iloc[i]['turns median'],df7a.iloc[i]['Map']),xytext=(-50,5),textcoords ='offset points',color='b')
            ax.annotate(int(df7a.iloc[i]['turns count']),(df7a['turns 10 pct'].min(),df7a.iloc[i]['Map']),xytext=(-37,2),textcoords ='offset points',color='b')
        ax.set_title('Arknova Map Ranking 10%-90%')
        ax.set_xlabel('turns')
        ax.grid(visible=True)
        ax.legend(loc="lower right")
        ax.set_xticks(range(int(ax.get_xlim()[0]),int(ax.get_xlim()[1])+1))
    else:
        if (mymap != '0') & (mymap !=  'A'):
            df2 = df[df['Map'].str.contains(' ' + mymap + ':')].groupby('turns').agg({'count':'sum'}).reset_index()
            ax.set_title(f"Arknova {df[df['Map'].str.contains(' ' + mymap + ':')]['Map'].tolist()[0]}")
        else:
            df2 = df[df['Map'] == 'Map ' + mymap].groupby('turns').agg({'count':'sum'}).reset_index()
            ax.set_title(f"Arknova {df[df['Map'] == 'Map ' + mymap]['Map'].tolist()[0]}")
        df2.plot.scatter(x='turns',y='count',color='b',label=f'Elo {mymin}-{mymax}',ax=ax,alpha=0.2)
        ax.set_xticks(np.arange(round(ax.get_xlim()[0],0),round(ax.get_xlim()[1],0)+1))
        width = 950 / len(np.arange(round(ax.get_xlim()[0],0),round(ax.get_xlim()[1],0)+1))
        df2 = df2.sort_values(['turns'])
        df2['cs']=round(np.cumsum(df2['count']) / df2['count'].sum() * 100,1)
        yo = 5
        for i in range(len(df2)):
            line1 = Line2D([df2.iloc[i]['turns'],df2.iloc[i]['turns']],[0,df2.iloc[i]['count']],color='b',linewidth=width,alpha=0.2)
            ax.add_line(line1)
            ax.annotate(df2.iloc[i]['cs'].astype('str') + '%',(df2.iloc[i]['turns'],0),xytext=(-13,yo),textcoords ='offset points',color='k', weight='bold')
            if yo == 5:
                yo = 15
            else:
                yo = 5
        ax.set_xlabel('turns')
        ax.set_ylabel('count')
        ax.set_ylim(0,ax.get_ylim()[1])
        ax.grid(visible=True)
        ax.legend(loc="upper right")
else:
    if mymap == 'all':
        df6a.plot.scatter(y='Map',x='perturn median',ax=ax,color='b',s=100,label=f'Elo {mymin}-{mymax}')
        df6a.plot.scatter(y='Map',x='perturn 10 pct',ax=ax,color='b',marker='<',s=100)
        df6a.plot.scatter(y='Map',x='perturn 90 pct',ax=ax,color='b',marker='>',s=100)
        df6a.plot.scatter(y='Map',x='perturn mean',ax=ax,color='b',marker='^',s=100)
        for i in range(len(df6a)):
            line1 = Line2D([df6a.iloc[i]['perturn 10 pct'],df6a.iloc[i]['perturn 90 pct']],[df6a.iloc[i]['Map'],df6a.iloc[i]['Map']],color='b',linewidth=10,alpha=0.2)
            ax.add_line(line1)
#            ax.annotate(int(df6a.iloc[i]['perturn count']),(df6a.iloc[i]['perturn median'],df6a.iloc[i]['Map']),xytext=(5,5),textcoords ='offset points',color='b')
            ax.annotate(int(df6a.iloc[i]['perturn count']),(df6a['perturn 90 pct'].min(),df6a.iloc[i]['Map']),xytext=(-37,2),textcoords ='offset points',color='b')
        ax.set_title('Arknova Map Ranking 10%-90%')
        ax.set_xlabel('points/turn')
        ax.grid(visible=True)
        ax.legend(loc="upper right")
        ax.set_xticks(range(int(ax.get_xlim()[0]),int(ax.get_xlim()[1])+1))
    else:
        if (mymap != '0') & (mymap !=  'A'):
            df2 = df[df['Map'].str.contains(' ' + mymap + ':')].groupby('perturn').agg({'count':'sum'}).reset_index()
            ax.set_title(f"Arknova {df[df['Map'].str.contains(' ' + mymap + ':')]['Map'].tolist()[0]}")
        else:
            df2 = df[df['Map'] == 'Map ' + mymap].groupby('perturn').agg({'count':'sum'}).reset_index()
            ax.set_title(f"Arknova {df[df['Map'] == 'Map ' + mymap]['Map'].tolist()[0]}")
        df2.plot.scatter(x='perturn',y='count',color='b',label=f'Elo {mymin}-{mymax}',ax=ax,alpha=0.2)
        ax.set_xticks(np.arange(round(ax.get_xlim()[0],1),round(ax.get_xlim()[1],1)+0.1,0.1))
        width = 950 / len(np.arange(round(ax.get_xlim()[0],1),round(ax.get_xlim()[1],1)+0.1,0.1))
        df2 = df2.sort_values(['perturn'],ascending=False)
        df2['cs']=round(np.cumsum(df2['count']) / df2['count'].sum() * 100,1)
        yo = 5
        for i in range(len(df2)):
            line1 = Line2D([df2.iloc[i]['perturn'],df2.iloc[i]['perturn']],[0,df2.iloc[i]['count']],color='b',linewidth=width,alpha=0.2)
            ax.add_line(line1)
            ax.annotate(df2.iloc[i]['cs'].astype('str') + '%',(df2.iloc[i]['perturn'],0),xytext=(-13,yo),textcoords ='offset points',color='k', weight='bold')
            if yo == 5:
                yo = 15
            else:
                yo = 5
        ax.set_xlabel('points/turn')
        ax.set_ylabel('count')
        ax.set_ylim(0,ax.get_ylim()[1])
        ax.grid(visible=True)
        ax.legend(loc="upper right")
buf = BytesIO()
fig.savefig(buf, format="png")
data = base64.b64encode(buf.getbuffer()).decode("ascii")
st.write(f"{(dt.now() - start).total_seconds()} seconds<img src='data:image/png;base64,{data}'/>",unsafe_allow_html=True)
