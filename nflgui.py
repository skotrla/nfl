import sqlite3
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype)
import pandas as pd
import streamlit as st
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np
import warnings
import os

v = 1.0

warnings.filterwarnings("ignore")

id = 0

def filter_dataframe(df: pd.DataFrame, coll=[]) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    global id 
    id += 1
    modify = st.checkbox("Add filters", key=id)

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() <= 2 or column in coll:
                if df[column].nunique() <= 2:
                    user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()))
                else:
                    user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=[])
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                step = 0.5
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def join_files(input_files, output_file):
    with open(output_file, "wb") as outfile:
        for filename in input_files:
            with open(filename, "rb") as infile:
                outfile.write(infile.read())

#streamlit run nflgui.py --server.port=4016
st.set_page_config(layout="wide")
st.markdown("""
                <html>
                <style>
                        ::-webkit-scrollbar {
                            width: 2vw;
                            }

                            /* Track */
                            ::-webkit-scrollbar-track {
                            background: #f1f1f1;
                            }

                            /* Handle */
                            ::-webkit-scrollbar-thumb {
                            background: #888;
                            }

                            /* Handle on hover */
                            ::-webkit-scrollbar-thumb:hover {
                            background: #555;
                            }
                </style>
            """, unsafe_allow_html=True)

db = st.query_params.get_all('db')
if len(db)==0:
    db.append('')

#match db[0]:
#    case 'alt':
#        pass
#    case _:
if db[0]=='':
        #connection = sqlite3.connect('c://users//2019//desktop//print//nfl.db')
        #connection2 = sqlite3.connect('c://users//2019//desktop//print//nfl2.db')
        #nfl = pd.read_sql(f'SELECT g1.* FROM games g1 INNER JOIN (SELECT Week, Year, RTeamN, MAX(Date) as Date FROM games GROUP BY Week, Year, RTeamN) g2 ON g1.Week=g2.Week AND g1.Year=g2.Year AND g1.RTeamN=g2.RTeamN AND g1.Date=g2.Date',connection).drop(columns=['index'])
        #lastdate = pd.read_sql(f'SELECT Max(Date) as Date FROM games',connection)['Date'].tolist()[0]
        connection = sqlite3.connect('nfl.db')
        connection2 = sqlite3.connect('nfl2.db')
        nfl = pd.read_sql(f'SELECT g1.* FROM games g1 INNER JOIN (SELECT Week, Year, RTeamN, MAX(Date) as Date FROM games GROUP BY Week, Year, RTeamN) g2 ON g1.Week=g2.Week AND g1.Year=g2.Year AND g1.RTeamN=g2.RTeamN AND g1.Date=g2.Date',connection).drop(columns=['index'])
        nflb = pd.read_sql(f'SELECT g1.* FROM games g1 INNER JOIN (SELECT Week, Year, RTeamN, MAX(Date) as Date FROM games GROUP BY Week, Year, RTeamN) g2 ON g1.Week=g2.Week AND g1.Year=g2.Year AND g1.RTeamN=g2.RTeamN AND g1.Date=g2.Date',connection2).drop(columns=['index'])
        nflc = pd.concat([nfl,nflb])
        nflb = nflc.groupby(['Week','Year','RTeamN']).agg({'Date':'max'}).reset_index()
        nfl = nflc.merge(nflb,how='inner',on=['Week','Year','RTeamN','Date'])        
        lastdate = pd.read_sql(f'SELECT * from lastdate',connection2)
        lastdate['Date'] = pd.to_datetime(lastdate['Date'],format='mixed')
        lastdate = lastdate['Date'].tolist()[0]
        nfl['Date']=pd.to_datetime(nfl['Date'],format='mixed')
        nfl['GameDate']=pd.to_datetime(nfl['GameDate'],format='mixed') - td(hours=6)
        nfl['SCutoff'] = nfl['SCutoff'].astype('float')        
        nfl['OCutoff'] = nfl['OCutoff'].astype('float')        
        coll = nfl.columns
        nfl['TActual'] = nfl['HActual'] + nfl['RActual']
        nfl['SActual'] = np.ceil(abs((nfl['HActual']-nfl['RActual'])/8))
        nfl['Winner'] = np.where(nfl['HActual'] > nfl['RActual'],nfl['HTeamN'],nfl['RTeamN'])
        nfl['Winner'] = np.where(nfl['HActual'] == nfl['RActual'],'Tie',nfl['Winner'])
        nfl['Loser'] = np.where(nfl['HActual'] < nfl['RActual'],nfl['HTeamN'],nfl['RTeamN'])
        nfl['Loser'] = np.where(nfl['HActual'] == nfl['RActual'],'Tie',nfl['Loser'])
        fdf = filter_dataframe(nfl,[])
        st.title('NFL Game Most Recent Model/Spread')
        tactual = fdf['TActual'].min()
        scutoff = fdf['SCutoff'].min()
        ocutoff = fdf['OCutoff'].min()
        if tactual > 0:
            if ocutoff >= 3:
                owin = round(fdf['OWin'].mean()*100)
                st.title(f'Over Model Accuracy: {owin}%')
            if scutoff >= 3:
                swin = round(fdf['SWin'].mean()*100)
                st.title(f'Spread Model Accuracy: {swin}%')
        #st.data_editor(
        #    fdf,
        #    column_config={
        #   "Week": st.column_config.NumberColumn(format="%d"),
        #    "Year": st.column_config.NumberColumn(format="%d"),
        #    hide_index=True)
        st.dataframe(fdf, use_container_width=True,hide_index=True)
        st.markdown(f'<i>{len(fdf)} rows out of {len(nfl)} total rows<br>Last updated: {lastdate} CT</i>',unsafe_allow_html=True)
        st.title('NFL Game Details')
        nfl2 = pd.read_sql(f'SELECT * FROM games',connection).drop(columns='index')
        nfl2b = pd.read_sql(f'SELECT * FROM games',connection2).drop(columns='index')
        nfl2 = pd.concat([nfl2,nfl2b])
        nfl2['Date']=pd.to_datetime(nfl2['Date'],format='mixed')
        nfl2['GameDate']=pd.to_datetime(nfl2['GameDate'],format='mixed') - td(hours=6)
        nfl2 = nfl2.merge(fdf[['Year','Week','RTeamN']],how='inner',on=['Year','Week','RTeamN'])
        connection.close()
        fdf2 = filter_dataframe(nfl2,[])
        st.dataframe(fdf2, use_container_width=True,hide_index=True)
        st.markdown(f'<i>{len(fdf2)} rows out of {len(nfl)} total rows<br>Last updated: {lastdate} CT</i>',unsafe_allow_html=True)
        winner = fdf.groupby(['Winner']).agg({'Loser':'count'}).reset_index().rename(columns={'Winner':'Team','Loser':'Win Count'})
        loser = fdf.groupby(['Loser']).agg({'Winner':'count'}).reset_index().rename(columns={'Loser':'Team','Winner':'Loss Count'})
        winner = winner.merge(loser,how='outer',on='Team').fillna(0)
        minyear = fdf['Year'].min()
        maxyear = fdf['Year'].max()
        minscores = int(fdf['SActual'].min())
        fdf3 = filter_dataframe(winner,[])
        st.title(f'NFL Game Counts {minscores}+ Scores {minyear}-{maxyear}')
        st.dataframe(fdf3, use_container_width=False,hide_index=True)
        winner2 = fdf.groupby(['Winner','Year']).agg({'Loser':'count'}).reset_index().rename(columns={'Winner':'Team','Loser':'Win Count'})
        loser2 = fdf.groupby(['Loser','Year']).agg({'Winner':'count'}).reset_index().rename(columns={'Loser':'Team','Winner':'Loss Count'})
        winner2 = winner2.merge(loser2,how='outer',on=['Team','Year']).fillna(0)
        fdf4 = filter_dataframe(winner2,[])
        st.title(f'NFL Game Counts {minscores}+ Scores {minyear}-{maxyear}')
        st.dataframe(fdf4, use_container_width=False,hide_index=True)
if db[0]=='an':
#        connection = sqlite3.connect('c://users//2019//desktop//print//bga.db')
#        os.remove('bga.db')
        flist = [x for x in os.listdir('.') if x.find('bga.db') >= 0]
        if len(flist) == 0:
            flist = [x for x in os.listdir('.') if x.find('bgadb') >= 0]
            flist.sort()
#            os.system('cat ' + ' '.join(flist) + ' > bga.db')
            join_files(flist, 'bga.db')
        connection = sqlite3.connect('bga.db')        
        connection2 = sqlite3.connect('bga2.db')    
        bga = pd.read_sql(f'SELECT * FROM arknova', connection).drop(columns=['index'])
        bgab = pd.read_sql(f'SELECT * FROM arknova', connection2).drop(columns=['index'])
        bga = pd.concat([bga,bgab])
#        bga['Date'] = pd.to_datetime(bga['Date']).dt.strftime('%Y-%m-%d')
        bga['Date'] = pd.to_datetime(bga['Date'],format='mixed')
        bga = bga.sort_values(['Date'],ascending=False)
        bga['score']=bga['score'].astype('float')
        winner = bga.groupby(['table']).agg({'score':'max'}).reset_index().rename(columns={'score':'max'})
        bga = bga.merge(winner,how='inner',on='table')
        bga['winner']=np.where(bga['score']==bga['max'],True,False)
        bga = bga.drop(columns='max')    
#        lastdate = bga['Date'].max()
        lastdate = pd.read_sql(f'SELECT MAX(Date) as Date FROM lastdate',connection2)['Date'].tolist()[0]
        connection.close()
        connection2.close()
        coll = bga.columns
        fdf = filter_dataframe(bga,['player'])
        st.title('Ark Nova Stats '+ str(v))
        #st.data_editor(
        #    fdf,
        #    column_config={
        #   "Week": st.column_config.NumberColumn(format="%d"),
        #    "Year": st.column_config.NumberColumn(format="%d"),
        #    hide_index=True)
        st.dataframe(fdf, use_container_width=True,hide_index=True)
        st.markdown(f'<i>{len(fdf)} rows out of {len(bga)} total rows<br>Last updated: {lastdate}</i>',unsafe_allow_html=True)
if db[0]=='bga':
#       connection = sqlite3.connect('c://users//2019//desktop//print//bga.db')
        flist = [x for x in os.listdir('.') if x.find('bga.db') >= 0]
        if len(flist) == 0:
            flist = [x for x in os.listdir('.') if x.find('bgadb') >= 0]
            flist.sort()
#            os.system('cat ' + ' '.join(flist) + ' > bga.db')
            join_files(flist, 'bga.db')
        connection = sqlite3.connect('bga.db')
        connection2 = sqlite3.connect('bga2.db')    
#        bga = pd.read_sql(f'SELECT g.*, p.name FROM (SELECT * FROM games WHERE player IN (SELECT player FROM players WHERE pri=1)) g INNER JOIN players p ON g.player=p.player', connection)
#        bga['Date'] = pd.to_datetime(bga['Date'])
        pl = pd.read_sql(f'SELECT player FROM players WHERE pri=1',connection)
        plb = pd.read_sql(f'SELECT player FROM players WHERE pri=1',connection2)
        pl = pd.concat([pl,plb]).drop_duplicates()['player'].tolist()
        gt = pd.read_sql(f'SELECT * FROM games WHERE player IN ({str(pl)[1:-1]})',connection)
        gtb = pd.read_sql(f'SELECT * FROM games WHERE player IN ({str(pl)[1:-1]})',connection2)
        gt = pd.concat([gt,gtb]).drop_duplicates()
        pt = pd.read_sql(f'SELECT player,name FROM players',connection)
        ptb = pd.read_sql(f'SELECT player,name FROM players',connection2)
        pt = pd.concat([pt,ptb]).drop_duplicates()
        bga = gt.merge(pt,how='inner',on='player')
        bga = bga.sort_values(['table'],ascending=False)
        bga = bga.drop(columns=['index','date','elo2','elo change2','player'])
        bga2 = pd.read_sql(f'SELECT "table",Date,player as name FROM arknova', connection)
        bga2b = pd.read_sql(f'SELECT "table",Date,player as name FROM arknova', connection2)
        bga2 = pd.concat([bga2,bga2b])
        bga = bga.merge(bga2,how='left',on=['table','name'])
        bga['table'] = bga['table'].astype('int')
        bga['elo'] = bga['elo'].str.replace('mer','0').astype('float')
        bga['elo change'] = bga['elo change'].str.replace('mer','0').astype('float')        
#        bga['Date'] = pd.to_datetime(bga['Date']).dt.strftime('%Y-%m-%d')
        bga['Date'] = pd.to_datetime(bga['Date'],format='mixed')
        bga['date_delta'] = (bga['Date'] - bga['Date'].min())  / np.timedelta64(1,'D')
        bga.set_index('table', inplace=True)
        bga['date_delta'].interpolate(method='index', limit_direction='both', inplace=True)
        bga.reset_index(inplace=True)
        bga['Date'] = np.where(bga['Date'].isnull(), pd.to_datetime((bga['Date'].min() + pd.TimedeltaIndex(bga['date_delta'], unit='D')).astype('str').str[:10]),bga['Date'])
        bga = bga.drop(columns=['date_delta'])
#        dates = pd.read_sql(f'SELECT * FROM arknova', connection).drop(columns=['index'])
#        datesb = pd.read_sql(f'SELECT * FROM arknova', connection2).drop(columns=['index'])
#        dates = pd.concat([dates,datesb]).drop_duplicates()
#        dates['Date'] = pd.to_datetime(dates['Date']).dt.strftime('%Y-%m-%d')
#        dates['Date'] = pd.to_datetime(dates['Date'])
#        lastdate = dates['Date'].max()        
        lastdate = pd.read_sql(f'SELECT MAX(Date) as Date FROM lastdate',connection2)['Date'].tolist()[0]
        connection.close()
        connection2.close()
        coll = bga.columns
        fdf = filter_dataframe(bga,['name'])
        st.title('BGA Stats '+ str(v))
        #st.data_editor(
        #    fdf,
        #    column_config={
        #   "Week": st.column_config.NumberColumn(format="%d"),
        #    "Year": st.column_config.NumberColumn(format="%d"),
        #    hide_index=True)
        st.dataframe(fdf, use_container_width=True,hide_index=True)
        st.markdown(f'<i>{len(fdf)} rows out of {len(bga)} total rows<br>Last updated: {lastdate}</i>',unsafe_allow_html=True)
