ALLOWED_GENDERS={'Male','Female','Other'}
ALLOWED_MEMBERSHIP={'Basic','Silver','Gold','Platinum'}
ALLOWED_DEVICES={'Mobile','Desktop','Tablet'}
def num(v,name,lo,hi):
    try: x=float(v)
    except: raise ValueError(f'{name} must be numeric.')
    if not lo<=x<=hi: raise ValueError(f'{name} must be between {lo} and {hi}.')
    return x
def validate_customer_input(p):
    req=['age','income','website_visits','previous_purchases','avg_session_minutes','pages_viewed','email_clicks','support_tickets','days_since_last_purchase','membership_years','gender','membership_type','device']
    miss=[x for x in req if x not in p]
    if miss: raise ValueError('Missing fields: '+', '.join(miss))
    out={
      'age':num(p['age'],'age',18,100),'income':num(p['income'],'income',0,2000000),'website_visits':num(p['website_visits'],'website_visits',0,500),
      'previous_purchases':num(p['previous_purchases'],'previous_purchases',0,500),'avg_session_minutes':num(p['avg_session_minutes'],'avg_session_minutes',0,1000),
      'pages_viewed':num(p['pages_viewed'],'pages_viewed',0,1000),'email_clicks':num(p['email_clicks'],'email_clicks',0,500),
      'support_tickets':num(p['support_tickets'],'support_tickets',0,500),'days_since_last_purchase':num(p['days_since_last_purchase'],'days_since_last_purchase',0,3650),
      'membership_years':num(p['membership_years'],'membership_years',0,100)}
    for k,allowed in [('gender',ALLOWED_GENDERS),('membership_type',ALLOWED_MEMBERSHIP),('device',ALLOWED_DEVICES)]:
        if p[k] not in allowed: raise ValueError(f'{k} must be one of: {sorted(allowed)}')
        out[k]=p[k]
    return out
