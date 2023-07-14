from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle, json
import plotly.graph_objects as go
import plotly
import plotly.utils

app = Flask(__name__, template_folder='templates')



# auto division un-pickiling and formatting
#with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\auto_merge.pkl', 'rb') as auto:
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\Auto_merge1.pkl', 'rb') as auto:
    # Load the pickled object
    auto_dict = pickle.load(auto)
    auto_dict['actual_products'] = auto_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    auto_dict['most_similar_items'] = auto_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    auto_dict['recommend'] = auto_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    auto_dict['description'] = "As the product(s)" + " " + auto_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                               + " " + auto_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                               auto_dict['recommend']
    auto_dict = auto_dict[['customernumber', 'description','sim_score']]

# construction division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\Construction_merge1.pkl', 'rb') as construction:
    # Load the pickled object
    construction_dict = pickle.load(construction)
    construction_dict['actual_products'] = construction_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    construction_dict['most_similar_items'] = construction_dict['most_similar_items'].apply(
        lambda x: ", ".join(map(str, x)))
    construction_dict['recommend'] = construction_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    construction_dict['description'] = "As the product(s)" + " " + construction_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                                       + " " + construction_dict[
                                           'most_similar_items'] + " " + "which forms the bundle" + " " + \
                                       construction_dict['recommend']
    construction_dict = construction_dict[['customernumber', 'description']]

# cargo division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\Cargo_merge1.pkl', 'rb') as cargo:
    # Load the pickled object
    cargo_dict = pickle.load(cargo)
    cargo_dict['actual_products'] = cargo_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    cargo_dict['most_similar_items'] = cargo_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    cargo_dict['recommend'] = cargo_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    cargo_dict['description'] = "As the product(s)" + " " + cargo_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                                + " " + cargo_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                                cargo_dict['recommend']
    cargo_dict = cargo_dict[['customernumber', 'description']]

# metal division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\Metal_merge1.pkl', 'rb') as metal:
    # Load the pickled object
    metal_dict = pickle.load(metal)
    metal_dict['actual_products'] = metal_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    metal_dict['most_similar_items'] = metal_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    metal_dict['recommend'] = metal_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    metal_dict['description'] = "As the product(s)" + " " + metal_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                                + " " + metal_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                                metal_dict['recommend']
    metal_dict = metal_dict[['customernumber', 'description']]

# Constr Project division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\ConstrSite project_merge1.pkl', 'rb') as cp:
    # Load the pickled object
    cp_dict = pickle.load(cp)
    cp_dict['actual_products'] = cp_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    cp_dict['most_similar_items'] = cp_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    cp_dict['recommend'] = cp_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    cp_dict['description'] = "As the product(s)" + " " + cp_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                             + " " + cp_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                             cp_dict['recommend']
    cp_dict = cp_dict[['customernumber', 'description']]

# house Engg division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\House Engineering_merge1.pkl', 'rb') as he:
    # Load the pickled object
    he_dict = pickle.load(he)
    he_dict['actual_products'] = he_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    he_dict['most_similar_items'] = he_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    he_dict['recommend'] = he_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    he_dict['description'] = "As the product(s)" + " " + he_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                             + " " + he_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                             he_dict['recommend']
    he_dict = he_dict[['customernumber', 'description']]

# Engg Workshop division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\Engineering workshop (Betriebswerkstatt)_merge1.pkl', 'rb') as ew:
    # Load the pickled object
    ew_dict = pickle.load(ew)
    ew_dict['actual_products'] = ew_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    ew_dict['most_similar_items'] = ew_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    ew_dict['recommend'] = ew_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    ew_dict['description'] = "As the product(s)" + " " + ew_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                             + " " + ew_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                             ew_dict['recommend']
    ew_dict = ew_dict[['customernumber', 'description']]

# wood division un-pickiling and formatting
with open(r'C:\Users\wn00210780\PycharmProjects\pythonProject2\Wood_merge1.pkl', 'rb') as wood:
    # Load the pickled object
    wood_dict = pickle.load(wood)
    wood_dict['actual_products'] = wood_dict['actual_products'].apply(lambda x: ", ".join(map(str, x)))
    wood_dict['most_similar_items'] = wood_dict['most_similar_items'].apply(lambda x: ", ".join(map(str, x)))
    wood_dict['recommend'] = wood_dict['recommend'].apply(lambda x: ", ".join(map(str, x)))
    wood_dict['description'] = "As the product(s)" + " " + wood_dict[
        'actual_products'] + " " + "is purchased often, we are suggesting these products" \
                               + " " + wood_dict['most_similar_items'] + " " + "which forms the bundle" + " " + \
                               wood_dict['recommend']
    wood_dict = wood_dict[['customernumber', 'description']]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        view = request.form.get('view')
        if view == 'admin':
            return redirect(url_for('admin_view'))
        elif view == 'user':
            return redirect(url_for('user_view'))
    return render_template('in1.html')


@app.route('/admin')
def admin_view():
    return render_template('index.html')


@app.route('/user', methods=['GET', 'POST'])
def user_view():
    if request.method == 'POST':
        option = request.form['option']
        return redirect(url_for('results', option=option))
    return render_template('user.html')


@app.route('/results/<option>')
def results(option):
    if f"{option}" == 'Construction':
        return render_template('div_recommend.html', table=construction_dict.to_dict('records'))
    elif f"{option}" == 'Metal':
        return render_template('div_recommend.html', table=metal_dict.to_dict('records'))
    elif f"{option}" == 'Auto':
        return render_template('div_recommend.html', table=auto_dict.to_dict('records'))
    elif f"{option}" == 'Cargo':
        return render_template('div_recommend.html', table=cargo_dict.to_dict('records'))
    elif f"{option}" == 'ConstrSite project':
        return render_template('div_recommend.html', table=cp_dict.to_dict('records'))
    elif f"{option}" == 'House Engineering':
        return render_template('div_recommend.html', table=he_dict.to_dict('records'))
    elif f"{option}" == 'Wood':
        return render_template('div_recommend.html', table=wood_dict.to_dict('records'))
    elif f"{option}" == 'Engineering workshop (Betriebswerkstatt)':
        return render_template('div_recommend.html', table=ew_dict.to_dict('records'))
    else:
        return 'not possible'


@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        option = request.form['option']
        return redirect(url_for('result', option=option))
    return render_template('options.html')


# @app.route('/page2', methods=['GET', 'POST'])
# def page2():
#     if request.method == 'POST':
#         selected_key = request.form['key']
#         return redirect(url_for('display_ncfvalue', key=selected_key))
#     else:
#         return render_template('ncf.html', keys=list(ncf_dict.keys()))


@app.route('/page2')
def page2():
    with open(r'C:\Users\wn00210780\PycharmProjects\visualizeflow\hyperparam.pkl', 'rb') as hp:
        # Load the pickled object
        hyper_param = pickle.load(hp)

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=hyper_param['latent_dim'],
                  colorscale=[[0, 'purple'], [0.15, 'lightseagreen'], [0.25, 'red'], [1, 'gold']]),
        dimensions=list([
            dict(range=[0, 150],
                 # constraintrange = [0,16],
                 label='Latent Dim', values=hyper_param['latent_dim']),
            dict(range=[-0.02, 0.15],
                 # constraintrange = [0,1],
                 label='Drop Out', values=hyper_param['drop_out']),
            dict(range=[-0.02, 0.10],
                 label='Regularization Value', values=hyper_param['reg_value']),
            dict(range=[0.00001, 0.01],
                 label='Learning Rate', values=hyper_param['learning_rate']),
            # dict(range=[0.00001, 256],
            #      label='Batch Size', values=hyper_param['batch_size']),
            dict(range=[0.00001, 0.5],
                 # constraintrange = [0.0001,0.1],
                 label='RMSE', values=hyper_param['RMSE']),
        ])
    )
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title='Hyperparameter Tuning Results'
    )

    # Convert the figure to a JSON object
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('ncf_hypervalue.html', graphJSON=graphJSON)


@app.route('/page3')
def page3():
    return render_template('ncfpred.html')#'NCF_Results.html')

@app.route('/page4')
def page4():
    return render_template('product_hierarchy.html')

@app.route('/page5')
def page5():
    return render_template('ncf_cluster_a.html')
# @app.route('/display_alsvalue/<key>')
# def display_alsvalue(key):
#     value = als_dict.get(key, 'Invalid key')
#     return render_template('NCF_Results.html', key=key, value=value)
@app.route('/page6')
def page6():
    return render_template('hiddenlayer_ncf.html')

@app.route('/page7')
def page7():
    return render_template('combined_view.html')


@app.route('/result/<option>')
def result(option):
    if f"{option}" == 'Construction':
        return render_template('Construction_association_rules.html')
    elif f"{option}" == 'Metal':
        return render_template('Metal_association_rules.html')
    elif f"{option}" == 'Auto':
        return render_template('Auto_association_rules.html')
    elif f"{option}" == 'Cargo':
        return render_template('cargo_association_rules.html')
    elif f"{option}" == 'ConstrSite project':
        return render_template('ConstrSite_project_association_rules.html')
    elif f"{option}" == 'House Engineering':
        return render_template('House_Engineering_association_rules.html')
    elif f"{option}" == 'Wood':
        return render_template('Wood_association_rules.html')
    elif f"{option}" == 'Engineering workshop (Betriebswerkstatt)':
        return render_template('Engineering_workshop_association_rules.html')
    else:
        return 'not possible'


if __name__ == '__main__':
    app.run(debug=True)
