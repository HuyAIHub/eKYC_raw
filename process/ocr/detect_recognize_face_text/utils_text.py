import re
from utils_ocr.address_correction import AddressCorrection


def fix_surname(text):
    """ auto-correct the surname in case meet special character(s)
    :param text: the predict surname
    :return: the fix surname
    """
    txat = text.split(" ")

    if len(txat) > 1:
        text_front = txat[0]
        """
        Thay the chu sai chinh ta trong phan ho
        """
        ho_nguyen = [[("ÑGUYẾẼN", "ÑÑQUYẼN", "NIGUYÉN", "ÑGUYẼN", "ÑGUYẼN", "ÑGUYỄN", "ÑGUYÊN", "NGUYÊÊN", "NGUYÊẼN",
                       "NGUYEN", "NGUYẼN", "NGUYẾN", "NGUYỆN", "NGUYỂN", "NGUYỀN", "NNGUYÊN", "NGUUỄN"), "NGUYỄN"]]
        ho_tran = [[("TRẤN", "TRẨN", "TRẬN", "TRẪN", "TRÂN", "TRÂÂN", "RÂN", "TTRẦN", "TTRẨN", "TRÀN", "TRẠN"), "TRẦN"]]
        ho_hoang = [[("HOANG", "HOẢNG", "HOẠNG", "HOÁNG", "HOÃNG"), "HOÀNG"]]
        ho_huynh = [[("HUÝNH", "HUỶNH", "HUỴNH", "HUỸNH"), "HUỲNH"]]
        ho_pham = [[("PHẢM", "PHÀM", "PHÁM", "PHÃM", "PIHẠM"), "PHẠM"]]
        # ho_ngo = [[("NGỒ"), "NGÔ"]]
        ho_vu = [[("VÙ", "VÚ", "VỤ", "VU"), "VŨ"]]
        ho_do = [[("ĐÔ", "ĐỐ", "ĐO", "DO", "ĐỒ"), "ĐỖ"]]
        ho_ho = [[("HỔ", "HÒ", "HỐ", "HỖ", "HÔ"), "HỒ"]]
        ho_doan = [[("ĐỖÀN", "ĐỖÀN"), "ĐOÀN"]]

        ho_vn = ho_nguyen + ho_tran + ho_hoang + ho_huynh + ho_pham + ho_ho + ho_vu + ho_do + ho_doan 
        for ho in ho_vn:
            for ho_sai in ho[0]:
                text_front = text_front.replace(ho_sai, ho[1])
        """
        Thay the khi phan ho bi khuyet 1 ki tu (khong dung ham 'replace')
        """
        ho_vn_khuyetchudau = {
            "GUYỄN": "NGUYỄN", "RẦN": "TRẦN", "OÀNG": "HOÀNG", "UỲNH": "HUỲNH", "HẠM": "PHẠM", "Ũ": "VŨ", "OÀN": "ĐOÀN"
        }
        if text_front in ho_vn_khuyetchudau.keys():
            text_front = ho_vn_khuyetchudau[text_front]

        """
        giu nguyen cac phan con lai
        """
        for txt in txat[1:]:
            text_front = text_front + " " + txt
        return text_front
    else:
        return text

def fix_year(text):
    """ auto-correct the year
    :param text: the predict year
    :return: the fix year
    """
    fix_text = re.sub("[^0123456789/-]", "", text)
    if len(text) == 3:  # case lost the 1st character. Ex: 017 -> 2017
        fix_text = '1' + text if text[0] == '9' else '2' + text if text[0] in ['0', '1'] else text
    elif len(text) == 4:  # case wrong the 1st or 2nd character.
        if text[0] == '1' and text[1] != '0':  # Ex: 1687 -> 1987
            fix_text = '19' + fix_text[2:]
        elif text[0] == '2' and text[1] != '9':  # Ex: 2637 -> 2037
            fix_text = '20' + fix_text[2:]
        elif text[0] != '2' and text[1] == '9':  # Ex: 7987 -> 1987
            fix_text = '19' + fix_text[2:]
        elif text[0] != '1' and text[1] == '0':  # Ex: 8037 -> 2037
            fix_text = '20' + fix_text[2:]
    elif len(text) >= 5:
        while True:
            if len(fix_text) >= 5 and fix_text[0] not in ['1', '2']:
                fix_text = fix_text[1:]
            else:
                break
        if len(fix_text) >= 5:
            fix_text = fix_text[:4]
    return fix_text


def fix_month(month):
    """ auto-correct the month
    :param month: the predict month
    :return: the fix month
    """
    fix_m = re.sub("[^0123456789/-]", "", month)
    if len(month) == 2:
        if month[1] in ['3', '4', '5', '6', '7', '8', '9']:
            fix_m = '0' + fix_m[1]
        elif month[0] != '0':  # month[1] in ['0', '1', '2']
            fix_m = '1' + fix_m[1]
    elif len(month) >= 3:
        while True:
            if len(fix_m) >= 3 and fix_m[0] not in ['0', '1']:
                fix_m = fix_m[1:]
            else:
                break
        if len(fix_m) >= 3:
            fix_m = fix_m[:2]
    return fix_m


def fix_day(day):
    """ auto-correct the day
    :param day: the predict day
    :return: the fix day
    """
    fix_d = re.sub("[^0123456789/-]", "", day)
    if len(day) >= 3:
        while True:
            if len(fix_d) >= 3 and fix_d[0] not in ['0', '1', '2', '3']:
                fix_d = fix_d[1:]
            else:
                break
        if len(fix_d) >= 3:
            fix_d = fix_d[:2]
    return fix_d

def fix_datetime(text):
    """ auto-correct the expired date in case wrong format. Ex: 29/122028 -> 29/12/2028
    :param text: the predict 'expired date'
    :return: the fix 'expired date'
    """
    legal_yy = ['19', '20']
    fix_text = re.sub("[^0123456789/-]", "", text)
    # pr(__file__, "fix_datetime() fix_text = '{}'".format(fix_text))
    fix_text = fix_text.lstrip()
    if len(fix_text) > 0 and fix_text[0] not in "0123456789":
        fix_text = fix_text[1:]
    fix_text = fix_text.rstrip()

    if fix_text.find('/') != -1:
        character = '/'
    elif fix_text.find('-') != -1:
        character = '-'
    else:  # khong tim thay dau ngan cach '/' hoac '-'
        yy = fix_text[-4:-2]  # hai so dau cua nam, phai la '19' hoac '20'
        if yy in legal_yy:
            if len(fix_text) == 6:  # dmyyyy
                fix_text = fix_text[0] + '/' + fix_text[1] + '/' + fix_text[2:]
            elif len(fix_text) == 8:  # ddmmyyyy
                fix_text = fix_text[:2] + '/' + fix_text[2:4] + '/' + fix_text[4:]
            elif len(fix_text) == 7:  # ddmyyyy hoac dmmyyyy
                if fix_text[0] in ['4', '5', '6', '7', '8', '9']:  # dmmyyyy
                    fix_text = fix_text[:1] + '/' + fix_text[1:3] + '/' + fix_text[3:]
                elif fix_text[1] not in ['0', '1']:  # ddmyyyy
                    fix_text = fix_text[:2] + '/' + fix_text[2] + '/' + fix_text[3:]
                elif fix_text[1] == '1' and fix_text[2] not in ['0', '1', '2']:  # ddmyyyy
                    fix_text = fix_text[:2] + '/' + fix_text[2] + '/' + fix_text[3:]
                elif fix_text[2] == '0':  # dmmyyyy
                    fix_text = fix_text[:1] + '/' + fix_text[1:3] + '/' + fix_text[3:]
            return fix_text
        else:
            return fix_text

    dt_nums = fix_text.split(character)
    # print("len(dt_nums) = {}".format(len(dt_nums)))
    if len(dt_nums) == 2:  # truong hop thieu mot dau ngan cach '/' hoac '-'
        if len(dt_nums[1]) == 4:  # VD dm/yyyy hoac ddm/yyyy hoac dmm/yyyy hoac ddmm/yyyy
            if len(dt_nums[0]) == 2:
                fix_text = fix_text[:1] + character + fix_text[1:]
            elif len(dt_nums[0]) == 4:
                fix_text = fix_text[:2] + character + fix_text[2:]
            elif len(dt_nums[0]) == 3:
                if dt_nums[0][0] in ['4', '5', '6', '7', '8', '9']:
                    fix_text = fix_text[:1] + character + fix_text[1:]
                elif dt_nums[0][1] not in ['0', '1'] or dt_nums[0][2] not in ['0', '1', '2']:
                    fix_text = fix_text[:2] + character + fix_text[2:]
                elif dt_nums[0][2] == '0':
                    fix_text = fix_text[:1] + character + '1' + fix_text[2:]
        else:  # VD dd/myyyy
            fix_text = fix_text[:-4] + character + fix_text[-4:]
        # xu li them cho phan nam '19' hoac '20'
        fix_text = fix_text[:-4] + fix_year(fix_text[-4:])
    elif len(dt_nums) == 3:  # truong hop co du 2 dau ngan cach '/' hoac '-'
        # xu li thang va nam
        fix_text = fix_day(dt_nums[0]) + character + fix_month(dt_nums[1]) + character + fix_year(dt_nums[2])
    elif len(dt_nums) == 4:  # truong hop co thua (3) dau ngan cach '/' hoac '-'
        if len(dt_nums[-1]) < 4:  # co ki tu ngan cach o phan nam, vi du y/yy
            # xu li thang va nam
            fix_text = fix_day(dt_nums[0]) + character + fix_month(dt_nums[1]) + character + fix_year(
                dt_nums[2] + dt_nums[3])
    return fix_text

def fix_name_address_2(text):
    """
    check địa chỉ dựa trên cơ sở dữ liệu gồm 12k6 xã huyện hành chính của Việt Nam
    """
    fix_text = text.lower()
    address_correction = AddressCorrection()
    address = address_correction.address_correction(fix_text)
    address_components = address_correction.address_extraction(address)
    address = address.title()
    return address, address_components