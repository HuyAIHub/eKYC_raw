def json_format(result):
    
    label = [
        'identCardType',
        'identCardNumber',
        'identCardName',
        'identCardBirthDate',
        'identCardGender',
        'identCardNation',
        'identCardEthnic',
        'identCardCountry',
        "identCardCountryCity",
        "identCardCountryDistrict",
        "identCardCountryWards",
        "identCardCountryStreet",
        'identCardAdrResidence',
        "identCardAdrResidenceCity",
        "identCardAdrResidenceDistrict",
        "identCardAdrResidenceWards",
        "identCardAdrResidenceStreet",
        'identCardIssueDate',
        "identCardExpireDate",
        "identCardIssuePlace"]
    result1 = {}
    result1["identCardType"] = ""
    result1['identCardNumber'] = ""
    result1['identCardName'] = ""
    result1['identCardBirthDate'] = ""
    result1['identCardNation'] = ""
    result1['identCardGender'] = ""
    result1['identCardCountry'] = ""
    result1["identCardCountryCity"] = ""
    result1["identCardCountryDistrict"] = ""
    result1["identCardCountryWards"] = ""
    result1["identCardCountryStreet"] = ""
    result1['identCardAdrResidence'] = ""
    result1["identCardAdrResidenceCity"] = ""
    result1["identCardAdrResidenceDistrict"] = ""
    result1["identCardAdrResidenceWards"] = ""
    result1["identCardAdrResidenceStreet"] = ""
    result1['identCardIssueDate'] = ""
    result1["identCardExpireDate"] = ""
    result1["identCardIssuePlace"] = ""

    for element in result:
        if element["label"] in label:
            result1[element["label"]] = element["value"]
    if result1["identCardType"] == "GIẤY CHỨNG MINH NHÂN DÂN" or result1["identCardType"] == "CHỨNG MINH NHÂN DÂN":
        result1['identCardEthnic'] = result1["identCardNation"]
        result1["identCardNation"] = ""
    
    if result1["identCardNumber"] == '':
        return {}
    print("Test",result1["identCardNumber"])
    if not result1["identCardNumber"].isnumeric():
        return {}
    if len(result1["identCardNumber"]) == 9 or len(result1["identCardNumber"])==12:
        return result1
    else:
        return {}

def json_passport_format(result):
    label = [
        'PassportBirthday',
        'PassportDateOfIssue',
        'PassportDateOfExpire',
        'PassportFullname',
        'PassportGender',
        'PassportIDCard',
        'PassportID',
        'PassportNationality',
        'PassportCode',
        'PassportType',
        'PassportPlaceOfBirth',
        'PassportPlaceOfIssue',
        'PassportRMZ',
    ]
    result1 = {}
    result1["PassportBirthday"] = ""
    result1['PassportFullname'] = ""
    result1['PassportDateOfIssue'] = ""
    result1['PassportDateOfExpire'] = ""
    result1['PassportGender'] = ""
    result1['PassportIDCard'] = ""
    result1['PassportID'] = ""
    result1['PassportNationality'] = ""
    result1["PassportCode"] = ""
    result1["PassportType"] = ""
    result1["PassportPlaceOfBirth"] = ""
    result1["PassportPlaceOfIssue"] = ""
    result1['PassportRMZ'] = ""
    for element in result:
        if element["label"] in label:
            result1[element["label"]] = element["value"]
    if result1["PassportID"] == '':
        return {}
    return result1