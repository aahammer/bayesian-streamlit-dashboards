import plotly.graph_objects as go

def create( num, avg, deal_chance):


    values = [100, 100 * avg[0], deal_chance * 100]
    labels = ["Pending Contacts", "Ongoing Meetings", "Deal Chance"]
    text_values = [f'{num[0]} pending <br> Conversion ~{avg[0]:.1%}'
        , f'{num[1]} ongoing <br> ~{avg[1]:.1%} conversion'
        , f'{deal_chance:.1%} chance <br> to close a deal']

    hover_texts = text_values

    fig = go.Figure(go.Funnel(
        x=labels,
        y=values,
        text=text_values,
        textposition=['inside' if v > 25 else 'outside' for v in values],
        textinfo="text",  # Display custom text and percentage
        texttemplate='%{text}',
        hovertext=hover_texts,
        hoverinfo="text",
        orientation='v',
        textfont={"size": 20},
        connector={"fillcolor": 'lightsteelblue'},
        marker={"color": ["mediumslateblue", "mediumslateblue",
                          f"{'mediumslateblue' if deal_chance * 100 > 90 else 'red'}"]}  # Colors for each funnel stage
    ))

    fig.update_layout(
        yaxis={'side': 'right'}
    )


    current_height = fig.layout.height or 400
    new_height = 0.8 * current_height

    fig.update_layout(height=new_height, margin=dict(t=20, b=0, l=0, r=0), xaxis=dict(tickfont=dict(size=20)))

    return fig