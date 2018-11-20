import xml.etree.ElementTree as ET

# Returns a set of all tags within a tree
# Adds current tag to set then recursively searches for more tags
def dive(root):
    n = set()
    for child in root:
        n.add(child.tag)
        x = dive(child)
        n = n.union(x)
    return n


# Tries to find the tag t anywhere within the root tree provided
# Returns first node whose tag is the desired one of False otherwise
def find_tag(root, t):
    if (root.tag == t):
        return root
    for child in root:
        ft = find_tag(child, t)
        if ft: return ft
    return False


# Tries to find the class c anywhere in the root tree provided
# Returns first node whose tag is the desired one or False otherwise
def find_class(root, c):
    if ('class' in root.attrib):
        if (root.attrib['class'] == c): return root
    for child in root:
        fc = find_class(child, c)
        if fc: return fc
    return False


# Finds all the text associated with a given root and ouputs a giant string
# Returns text of root along with text of all descendents
def text(root):
    txt = ''
    if root.text:
        txt = root.text.strip()
        if txt: return txt
        for child in root: txt += '\n' + text(child)
    return txt


def extract_text_from_xml(xml_file):

    # Generates a tree from the XML file of a specific articles and gets root
    doc = ET.parse(xml_file)
    root = doc.getroot()

    # Gets the path for the abstract and articles
    abstract = find_tag(root, 'abstract')
    fulltext = find_class(root, 'full_text')

    if abstract and fulltext:
        # If the abstracts and articles are non-empty path then we grab the
        # text in them
        abstract_string = text(abstract)
        fulltext_string = text(fulltext)

        return abstract_string, fulltext_string

    # Return None if the xml file was not correctly parsed
    return None, None
