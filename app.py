
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from pandas_profiling import ProfileReport
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import plotly.figure_factory as ff
from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.utils import calibration_and_holdout_data
import squarify
from dash import Dash, html, dcc
# TODO: from wordcloud import WordCloud

app = Dash(__name__)
server = app.server


from dash import Dash, html, dcc
from dash.dependencies import Input, Output

data = pd.read_excel("Dataset.xlsx")

data['InvoiceDate'].agg(['min', 'max'])

#images
image_a = 'images/a.png'
image_p = 'images/p.png'
#

fd = data.drop_duplicates()


# TODO:
# text = " ".join(review for review in data.Country.astype(str))
# x, y = np.ogrid[:300, :00]
# #mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
# #mask = 560 * mask.astype(int)
# wc = WordCloud(background_color="white", repeat=True, width=1600, height=800,  colormap='Dark2',)
# wc.generate(text)
# plt.axis("off")

# total purchase category plot
fd = fd[['Customer ID','Description','InvoiceDate','Invoice','Quantity','Price', 'Country']]
fd = fd[(fd['Quantity']>0)]
fd['TotalPurchase'] = fd['Quantity'] * fd['Price']

df_plot_bar = fd.groupby('Description').agg({'TotalPurchase':'sum'}).sort_values(by = 'TotalPurchase', ascending=False).reset_index().head(5)
df_plot_bar['Percent'] = round((df_plot_bar['TotalPurchase'] / df_plot_bar['TotalPurchase'].sum()) * 100,2)
fir_plotbar = px.bar(df_plot_bar, y='Percent', x='Description', title='Top selling products', 
text='Percent', color='Percent')
fir_plotbar.update_traces(texttemplate='%{text:.2s}', textposition='inside')
fir_plotbar.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(1, 0, 0, 0)',
})
fir_plotbar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',showlegend=False)

#  
df_plot = fd.groupby(['Country','Description','Price','Quantity']).agg({'TotalPurchase': 'sum'},{'Quantity':'sum'}).reset_index()
fig_miricle = px.scatter(df_plot[:25000], x="Price", y="Quantity", color = 'Country', 
        size='TotalPurchase',  size_max=20, log_y= True, log_x= True, title= "PURCHASE TREND ACROSS COUNTRIES")
fig_miricle.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(1, 0, 0, 0)',
})

#
time_serious_invoice = go.Figure([go.Scatter(x=fd['InvoiceDate'], y=fd['Quantity'])])

#
new = summary_data_from_transaction_data(fd, 'Customer ID', 'InvoiceDate', monetary_value_col='TotalPurchase', observation_period_end='2011-12-9')
new['percent'] = round((new['frequency'] / new['frequency'].sum()) * 100,2)
frequency_barchart = px.bar(new, y=new['percent'], x=new['frequency'], title='Frequency BarChart', color='percent')

#
fdg = fd.groupby(['Customer ID','Country']).agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                                 'Quantity': lambda quant: quant.sum(),
                                                 'Invoice': lambda num: len(num),
                                                 'TotalPurchase': lambda price: price.sum()    })

fdg.columns=['num_days','num_transactions','num_units','spent_money']
fdg['avg_order_value'] = fdg['spent_money']/fdg['num_transactions']
purchase_frequency = sum(fdg['num_transactions'])/4319
repeat_rate = round(fdg[fdg.num_transactions > 1].shape[0]/fdg.shape[0],2)
churn_rate = round(1-repeat_rate,2)


fdg['profit_margin'] = fdg['spent_money']*0.05
fdg['CLV'] = (fdg['avg_order_value']*purchase_frequency)/churn_rate
fdg.reset_index(inplace = True)
fdg['spent_money', 'avg_order_value','profit_margin'] = fdg.spent_money.apply(lambda x : "{:,}".format(x))


data.dropna(inplace=True)
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
data["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)
data["TotalPrice"] = data["Price"] * data["Quantity"]



rfm = data.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date- InvoiceDate.max()).days,
                                       "Invoice": lambda Invoice: Invoice.nunique(),
                                       "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
rfm.columns = ["recency","frequency","monetary"]
rfm = rfm[rfm["monetary"] > 0]
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

seg_map = {
    r'[1-2][1-2]': 'HIBERNATING',
    r'[1-2][3-4]': 'AT RISK',
    r'[1-2]5': 'CANT LOSE',
    r'3[1-2]': 'ABOUT TO SLEEP',
    r'33': 'NEED ATTENTION',
    r'[3-4][4-5]': 'LOYAL CUSTOMER',
    r'41': 'PROMISING',
    r'51': 'NEW CUSTOMERS',
    r'[4-5][2-3]': 'POTENTIAL LOYALIST',
    r'5[4-5]': 'CHAMPIONS'
}
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)


sgm= rfm["segment"].value_counts()
c = sgm.index.tolist()
d = sgm.tolist()
clv_chart = go.Figure(data=[go.Pie(labels=c, values=d, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])])
clv_chart.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(1, 0, 0, 0)',
})


df_treemap = rfm.groupby('segment').agg('count').reset_index()
fig_treemap = px.treemap(df_treemap, path=['segment'], values='RFM_SCORE')
fig_treemap.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(1, 0, 0, 0)',
})

# 
Total_Purchase = fd['Quantity'].sum()
Total_Customers = len(fd['Customer ID'].unique())
#


#
#BG/NBD Model 
cltv_df = data.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                           lambda date: (today_date - date.min()).days],
                                           'Invoice':      lambda num: num.nunique(),
                                           'TotalPrice':   lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
summary_cal_holdout = calibration_and_holdout_data(fd, 'Customer ID', 'InvoiceDate',
                                        calibration_period_end='2010-11-09',
                                      observation_period_end='2011-01-01',
                                                   monetary_value_col = 'TotalPurchase') 
summary_cal_holdout
summary_cal_holdout = summary_cal_holdout[(summary_cal_holdout['monetary_value_cal']>0)]

    
ggf.fit(summary_cal_holdout['frequency_cal'],
        summary_cal_holdout['monetary_value_cal'])
summary_cal_holdout['monetary_pred'] = ggf.conditional_expected_average_profit(summary_cal_holdout['frequency_holdout'],
                                         summary_cal_holdout['monetary_value_holdout'])


fig_scat = px.scatter(summary_cal_holdout, x="monetary_value_holdout", y="monetary_pred", width=400, labels={
                     "monetary_value_holdout": "Actual",
                     "monetary_pred": "Predicted"
                 },
                title="GG model prediction")


#
fig_d = px.density_contour(summary_cal_holdout, x="monetary_value_holdout", y="monetary_pred",  marginal_x="histogram", marginal_y="histogram")
#
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])

summary_cal_holdout['predicted_bgf'] = bgf.predict(30, 
                        summary_cal_holdout['frequency_cal'], 
                        summary_cal_holdout['recency_cal'], 
                        summary_cal_holdout['T_cal'])
fig_ggg = px.scatter(summary_cal_holdout, x="frequency_holdout", y="predicted_bgf", width=400, labels={
                     "frequency_holdout": "Actual",
                     "predicted_bgf": "Predicted"
                 },
                title="Beta Geo Fitter model prediction")

#
fig_o = px.density_contour(summary_cal_holdout, x="frequency_holdout", y="predicted_bgf",  marginal_x="histogram", marginal_y="histogram")
#
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False)
#hima gri
x = cltv_df['expected_average_profit'].values.tolist()
hist_data = [x]
group_labels = ['expected_average_profit'] # name of the dataset
fig_exp = ff.create_distplot(hist_data, group_labels)
#gri
cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12, 
                                   freq="W",  
                                   discount_rate=0.01)
cltv_12 = cltv_12.reset_index()
cltv_12 = cltv_df.merge(cltv_12, on="Customer ID", how="left")
cltv_12.sort_values(by="clv", ascending=False).head(5)

cltv_12_f = cltv_12.sort_values(by = 'clv', ascending=False).reset_index().head(50)
fig_miricle_0 = px.scatter(cltv_12_f, x='Customer ID', y="clv",  
        log_y= True, log_x= True, title= "Top 50 highest CLV customer predictions")
fig_miricle.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(1, 0, 0, 0)',
})
# gri
x = cltv_12['clv'].values.tolist()
cumsum = np.cumsum(x)
trace = go.Scatter(x=[i for i in range(len(cumsum))], y=10*cumsum/np.linalg.norm(cumsum),
                     marker=dict(color='rgb(150, 25, 120)'))
layout = go.Layout(
    title="CDF of 12 month clv predictions"
)
fig_sum = go.Figure(data=go.Data([trace]), layout=layout)
#










app = Dash(
    __name__,
    # TODO: implement styles
	external_stylesheets = [],
    suppress_callback_exceptions=True
)

server=app.server

app.layout = html.Div([
    html.Div([
		html.Div([
			html.H1('CLV Project')
		], id = 'title')
	]),
	html.Div([
		dcc.Tabs(id='tabs', value='tab-1', children=[
			dcc.Tab(label='CLV', value='clv-tab'),
			dcc.Tab(label='DATA DESCRIPTION', value='data-description-tab'),
			dcc.Tab(label='MODEL VISUALS', value='model-visuals-tab'),
    	]),
        html.Div(id='tab-content')
	]),
], className="container")

def render_stats():
    container_styles = {
        'display': 'flex',
        'flexWrap': 'wrap',
        'width': '100%',
    }

    item_styles = {
        'width': 'calc(50% - 128px)',
        'margin': '64px',
    }

    html.Div([
        html.Div([
            html.Span('Total Customers', children="Total Customers"),
            TotalCustomer,
            html.Span(children="10")
        ], style=item_styles),
        html.Div([
            html.Span('Total Purchases', children="Total Purchases"),
            TotalPurchases,
            html.Span(children="10")
        ], style=item_styles),
    ], className='stats-container', style=container_styles)

def render_clv_graphs():
    container_styles = {
        'display': 'flex',
        'flexWrap': 'wrap',
        'width': '100%',
    }


    item_styles = {
        'width': 'calc(50% - 128px)',
        'margin': '64px',
    }

    graphs = html.Div([
        html.Div([
            dcc.Graph(figure=clv_chart),
        ], style=item_styles),
        html.Div([
            html.Div([ 
            dcc.Graph(figure=fig_treemap)
            ]),
            html.Div([
                html.Div([
                    html.Img(src = app.get_asset_url('image_a'),
                     style = {'height': '30px'} 
                     )                
                ]),
                html.Div([
                     html.Img(src = app.get_asset_url('image_p'),
                     style = {'height': '30px'}
                     )
                ])
            ]),
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
        }),
    ], style=container_styles)

    return graphs

def render_data_description_graphs():
    container_styles = {
        'display': 'flex',
        'flexWrap': 'wrap',
        'width': '100%',
    }

    item_styles = {
        'width': 'calc(50% - 128px)',
        'margin': '64px',
    }

    graphs = html.Div([
        html.Div([
            dcc.Graph(figure=fir_plotbar),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_miricle),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=time_serious_invoice),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=frequency_barchart),
        ], style=item_styles),
    ], style=container_styles)

    return graphs

def render_model_visuals_graphs():
    container_styles = {
        'display': 'flex',
        'flexWrap': 'wrap',
        'width': '100%',
    }

    item_styles = {
        'width': 'calc(50% - 128px)',
        'margin': '64px',
    }

    item_styles3 = {
        'width': 'calc(80%)',
        'margin': '50px',
    }
     

    graphs = html.Div([
        html.Div([
            dcc.Graph(figure=fig_scat),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_d),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_ggg),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_o),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_sum),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_miricle_0),
        ], style=item_styles),
        html.Div([
            dcc.Graph(figure=fig_exp),
        ], style=item_styles3),        
    ], style=container_styles)
    

    return graphs

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'clv-tab':
        return html.Div([
            render_clv_graphs()
        ], id='data-description-tab')

    if tab == 'data-description-tab':
        return html.Div([
             html.Div([
                render_data_description_graphs()]),
        ], id='data-description-tab')

    if tab == 'model-visuals-tab':
        return html.Div([
             html.Div([
                render_model_visuals_graphs()]),
        ], id='data-description-tab')    


if __name__=='__main__':
	app.run_server(debug=True)
