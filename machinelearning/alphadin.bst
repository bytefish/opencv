%%  ALPHADIN.BST           Ausgabe [8.2]   2006-01-02
%%
%%  ersetzt ABBRVDIN.BST   Ausgabe [8.1b4]  15/12/2004
%%  wichtigste Aenderung gegenueber Version [8.1b4]:
%%  die Standardergaenzung "Online-Ressource" bei Internetquellen
%%  ist ersatzlos gestrichen worden (wegen ueberfluessiger Redundanz).
%%
%
%%  K.F.Lorenzen  (Copyright 1994-2006)  email:  lorenzen.marxen@t-online.de
%%  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%  ALPHADIN.BST wurde entwickelt aus Oren Patashnik's BibTeX standard
%%  bibliography style `alpha'. Eine vorgegebene Literaturdatenbank laesst sich
%%  somit beliebig nach Us- oder deutscher DIN 1505-Zitierkonvention
%%  verarbeiten.
%%  Analog zu den 4 US standard styles werden 4 DIN-gerechte bst-style Dateien
%%  veroeffentlicht (alphadin.bst, plaindin.bst, unsrtdin.bst, abbrvdin.bst).
%%  Die gueltige Version ist am schnellsten aus dem WWW ueber folgende URL zu
%%  beziehen:
%%            http://www.haw-hamburg.de/pers/Lorenzen/bibtex/
%%            Stand: 2006-01-02
%%  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%  WAS IST ALPHADIN.BST ?
%%  Dieser style produziert "deutsche" Literaturzitate in Literaturverzeichnis-
%%  sen gemaess der deutschen Norm DIN 1505, Teil 2 vom Jan. 1984.
%%  Die Literaturzitate werden alphabetisch nach Verfassern sortiert.
%%  Die Zitiermarken im Text und die Ordnungsmarken im Literaturverzeichnis
%%  bestehen aus abgekuerzten Verfasserbuchstaben plus Erscheinungsjahr in
%%  eckigen Klammern. Die vorliegenden DIN-styles gehen ueber einige bibliogra-
%%  phischen Beschraenkungen der Originalstyles hinaus.
%
%%  Es werden nun auch Elektronische Online / Offline Ressourcen wie Internet-
%%  quellen, CD-ROM usw. verarbeitet. Dazu kommen besondere Publikationsformen
%%  wie Patente, Normen, Karten, Fernsehaufzeichnungen, Gesetzesstellen, Spiele.
%
%%  Zur Gewaehrleistung der Allgemeingueltigkeit von bib-files gelten in den
%%  DIN-styles die in der Tex-Literatur veroeffentlichten originalen
%%  Definitionen und Regeln fuer die Publikationstypen (entry-types) und die
%%  bibliographischen Felder (fields).
%%  Treten kleinere "Schoenheitsfehler" im fertig gesetzten output auf,
%%  lassen sich diese so gut wie immer durch eine veraenderte
%%  Erfassung im bib-inputfile beseitigen. Oren Patashnik empfiehlt, die
%%  Definition der Felder weit auszulegen. Last but not least koennen
%%  Sie im output-file < *.bbl > noch letzte Hand zur Korrektur ansetzen.
%
%%  WARN- UND FEHLERMELDUNGEN
%%  Ursache von Warnmeldungen sind meistens ausgelassene Felder oder
%%  Erfassungs-"Fehler". Letztere haengen teilweise mit den gegenueber US-
%%  Gepflogenheiten andersartigen bibliographischen Regeln nach DIN 1505
%%  zusammen. Sie sind also in Wahrheit keine "Fehler" und duerfen fast immer
%%  ignoriert werden. Dennoch pruefen Sie diese Warnungen, um herauszufinden,
%%  ob Publikationstyp (=entry-type) und "fields" eventuell unzulaessig
%%  kombiniert worden sind.
%%  Echte Fehler ("errors") treten nur noch in Verbindung mit falscher
%%  Erfassung auf (nach meinen Tests!). Pruefen Sie die Syntax, den entry-type
%%  und die fields.
%%  Zu guter letzt: Qualitaetsmasstab ist einzig der DIN-konforme output!
%%
%%  DANKSAGUNG
%%  Hartmut Lueddecke, HAW Hamburg, hat viele Verbesserungsvorschlaege
%%  in die frueheren Versionen eingebracht. Ihm danke ich herzlich.
%%  Patrick W. Daly, dem Entwickler des Natbib-Stils, verdanke ich viele
%%  Anregungen und den steten Ansporn, die DIN-Stile zu verbessern.
%%  Viele an dieser Stelle ungenannt bleibende Anwender haben mich in
%%  den vergangenen Jahren auf Fehler oder Verbesserungsmoeglichkeiten
%%  aufmerksam gemacht und so diesen Stil mitentwickelt. Ihnen gilt mein
%%  besonderer Dank. Ihr Feedback ist immer willkommen und eine Ermunterung.
%%
%%  Klaus F. Lorenzen
%%
%%
%%
%%  HINWEIS: Neben den vier bibliographischen DIN 1505 Standard-Stilen
%%           nach den Original-styles von Oren Patashnik
%%           gibt es noch vier DIN 1505 Adaptionen der NATBIB-style
%%           Emulationen von Patrick W.Daly. Die Fuelle der darin
%%           implementierten Zitierweisen geht weit ueber die urspruenglichen
%%           Standardstyles hinaus.
%%           Naeheres unter URL http://www.haw-hamburg.de/pers/Lorenzen/bibtex/
%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%  version 0.99c for BibTeX versions 0.99c or later, LaTeX2e version
%%  Copyright (C) 1985, all rights reserved.
%%  Copying of this file is authorized only if either
%%  (1) you make absolutely no changes to your copy, including name, or
%%  (2) if you do make changes, you name it something other than
%%  alphadin.bst
%%  This restriction helps ensure that all standard styles are identical.
%%
%%
%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%  NEUE FELDER
%%  Zur Erfassung der neuen digitalen Online-Medien z.B. Internetquellen, E-journals, E-books,
%%  E-mail u.a. gibt es die zusaetzlichen Felder: doi, url, urn, lastchecked
%%  Internetquellen werden vorzugsweise mit dem BOOKLET-Typ erfasst.
%%  Normen, Patente, Schutzrechte sind mit dem MISC-Typ zu erfassen.

ENTRY
  { address      %%  Verlagsort (immer!)
    author       %%  persoenlicher Urheber eines Werkes oder am Zustandekommen
                 %%  beteiligte Personen(=Mitarbeiter, Uebersetzer, Redakteur u.a.)
    booktitle    %%  a) Gesamttitel eines mehrbaendigen Werkes
%                %%  b) Titel des Sammelwerks, das einzelne selbstaendige
%                %%     Beitraege mit eigenem Titel enthaelt ->incollection
    chapter      %%  Kapitel in einem Buch (Monographie)
    doi          %%% Digital Object Identifier ->article
    edition      %%  a) Auflagevermerk
%                %%  b) bei selbst. elektron. Quellen == Version ->booklet
    editor       %%  Persoenl.Herausgeber oder Koerperschaftlicher Herausgeber
    howpublished %%  beliebiger Verlegervermerk: veroeffentlicht "von wem, wo"
    institution  %%  Institution, die e.verlagsfreie Veroeffentlichung betreibt
    isbn         %%  Standardnr fuer Buecher
    issn         %%  - " -    :      Zeitschriften u. Serien
    journal      %%  Titel einer Zeitschrift
    key          %%  Zusaetzlich vergebener Sortierschluessel, mitunter notwend.
    lastchecked  %%  neues Feld fuer das Datum des Online-Abrufs
%                %%  einer Internetquelle (n. GRAY )
    month        %%  naehere Bestimmung des Erscheinungsjahres -> macro 's
    note         %%  freies Eingabefeld fuer zusaetzliche Informationen z. Quelle
    number       %%  Versch. Bedeutungen in Abhaengigkeit vom Eingabetyp:
%                %%  a) Bandnummer einer gezaehlten Reihe (series)
%                %%  b) Heftnummer einer Zeitschrift ->article
%                %%  c) Nummer eines Forschungsberichts ->techreport
    organization %%  a) Name der Organisation/des Organisators e. Tagung,Konferenz
%                %%  b) Name einer Firma/Gesellschaft, die ein ->manual herausgab
    pages        %%  Umfangsangaben, meist Seitenzahlen
    publisher    %%  Verlag
    school       %%  Hochschule/Universitaet, die eine Dipl.-Arb./Dissertation veroeff.
    series       %%  Titel e.Reihe, in der ein best. Buchtitel erschienen ist
    title        %%  Titel einer (namentlich gekennzeichneten) Veroeffentlichung
    type         %%  Zusatzfeld z.Kennzeichnung e.besonderen Publikationstyps
    url          %%  neues Feld URL ( Uniform Resource Locator ):
%                %%  Serveradresse einer Internetquelle
    urn          %%  neues Feld URN ( Uniform Resource Name ):
%                %%  Persistent Identifier einer Internetquelle
    volume       %%  a) Zaehlung bei einem mehrbaendigen Werk ->book/->proceedings
%                %%  b) Jahrgang einer Zeitschrift ->article
    year         %%  Erscheinungsjahr
  }
  {}
  { label extra.label sort.label short.list }

%%%----------------------------------------------------------------------------
%  Einige Standardvorgaben, die vom Benutzer veraendert werden koennen.
%%%----------------------------------------------------------------------------

%  Abkuerzung ("... und andere") bei Mehrverfasserquellen:

FUNCTION { ua.etal } { " u.\,a." } %% evtl. auch in eckigen Klammern " [u.\,a.]"

%%  oder lateinisch:  FUNCTION { ua.etal } { " et~al." }

FUNCTION { und } { " u. " }

%%  oder ausgeschrieben:  FUNCTION { und } { " und " }
%
% Einige elektronische Medien erhalten nach DIN 1505 eine "Ergaenzende Angabe"
% zusaetzlich zum materiellen Typ, z.B. CD ROM oder DVD u.a.:

FUNCTION { eress } { "Elektronische Ressource" }

%%%-----------------------------------------------------------------------------------

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

INTEGERS { colon.after  period.dash }

INTEGERS { after.firstblock }

INTEGERS { zahl  lang }

INTEGERS { nameptr namesleft numnames }

INTEGERS { ptr collation collrest }

INTEGERS { len }

INTEGERS { et.al.char.used }

INTEGERS { longest.label.width last.extra.num }
%-------------------------
STRINGS { longest.label last.sort.label next.extra }

STRINGS { h s t u v w }

STRINGS { fkt }
%-------------------------

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #4 'colon.after :=
  #5 'period.dash :=
  #6 'after.firstblock :=
}

FUNCTION {output.nonnull}
{ 's :=
  output.state after.block =
      { add.period$  write$
        newline$
        "\newblock " write$
      }
      { output.state before.all =
          { write$ }
          {  output.state colon.after =
               { ": " * write$
                 newline$
                 "\newblock " write$
               }
               { output.state period.dash =
                   { ". -- " *  write$
                     newline$
                     "\newblock " write$
                   }
                   { output.state mid.sentence =
                       { ", " * write$ }
                       { output.state after.sentence =
                           { " " * write$ }
                           { output.state after.firstblock =
                               { add.period$  write$
                                 newline$
                                 "\newblock " write$
                               }
                               { write$
                                 newline$
                                 "\newblock " write$
                               }
                             if$
                            }
                          if$
                       }
                     if$
                   }
                 if$
               }
             if$
           }
         if$
       after.block 'output.state :=
      }
    if$
    s
 }

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}


FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" write$
  label write$%
  "]{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ write$
  newline$
}

FUNCTION {set.period.dash}    %% Wenn ein ". - " die Satzteile trennen soll.!
{ output.state before.all =
    'skip$
    { period.dash  'output.state := }
  if$
}

FUNCTION {set.period.dash.check}
{ empty$
    'skip$
    'set.period.dash
  if$
}

FUNCTION {set.colon.after}   %%$$$ Wenn ein ": " d. Satzteile trennen soll!
{ output.state before.all =
    'skip$
    { colon.after 'output.state  := }
  if$
}

%%                                 Wenn ein " " die Satzteile trennen soll.!
FUNCTION {new.sentence}
{ output.state before.all =
     'skip$
     { after.sentence 'output.state := }
  if$
}

%%                                 Wenn ein ", " die Satzteile trennen soll.!
FUNCTION { part.of.sentence }
{ output.state before.all =
     'skip$
     { mid.sentence 'output.state := }
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\emph{" swap$ * "}" * }
  if$
}

FUNCTION { capitalize }
 { duplicate$ empty$
    { pop$ "" }
    { "\textsc{" swap$ * "}" * }
  if$
}


%% Adelspraedikate beim Sortieren nach den(m) Vornamen auffuehren
%% Abweichend v. DIN !!!
FUNCTION {format.names}
{ 's :=
  "" 'u  :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    {
       s nameptr "{ll}" format.name$ 't :=
       t capitalize 't :=
       s nameptr "{ jj}" format.name$ 'w :=
       s nameptr "{, ff}{ vv}{ jj}" format.name$ 'u :=
       u  text.length$  'lang :=
       #1 'zahl :=
       "" 'v :=
         {  zahl  lang  <  }
         {  u  zahl  #1  substring$  "~"  =
               {  v  ""  =
                     {  u  #1  zahl  #1  -  substring$  'v :=  }
                     'skip$
                  if$
                  v  u  zahl  #2  substring$ * "." *  w * 'v :=
                }
               'skip$
            if$
            zahl  #1  +  'zahl :=  }
      while$
      v  ""  =
         {  u  'v :=  }
         'skip$
      if$
      t  v  *  fkt  *  't :=
      nameptr #1 >
         { namesleft #1 >
             { " ; " * t * }
             { numnames #2 >
                  { " " * }
                  'skip$
               if$
               t "\textsc{others}" = t "\textsc{others} (Hrsg.)" = or
                  { ua.etal * }
                  { " ; " * t * }
               if$
               }
            if$  %% Ende der namesleft-Pruefung
          }
            't
           if$  %%  Ende der nameptr-Pruefung

      nameptr #1 + 'nameptr :=
      namesleft #1 -  'namesleft :=
    }
  while$  %%  Ende von { namesleft #0 > } ganz oben
  "" 'fkt :=  %% fkt wird zurueckgesetzt
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.editors}
{ editor empty$
   { author empty$
       { "empty author and editor in " cite$ * warning$ "" }
       'skip$
     if$
   }
   { author empty$
        { " (Hrsg.)" 'fkt :=
          editor format.names
        }
        { " (Hrsg.)" 'fkt :=
          " ; " * editor format.names *
        }
      if$
   }
 if$
 }

%%  2005-11-11
FUNCTION { format.authors.organization }
{ type$ "misc" =
        { organization empty$
            { author empty$
               { "" }
               { author format.names " (Erfinder)" * }
             if$
            }
            { author empty$
               { organization  }
               { author format.names " (Erfinder); " *
                 organization * " (Anmelder)" *
               }
             if$
            }
          if$
        }
        { type$ "manual" =
           { organization empty$
               { author empty$
                   { "" }
                   { author format.names }
                 if$
               }
               { author empty$
                   { organization capitalize " (Hrsg.)" * }
                   { author format.names " ; " *
                     organization capitalize * " (Hrsg.)" *
                   }
                 if$
               }
             if$
           }
           'skip$
         if$
        }
      if$
}


FUNCTION { format.editors.organization }
{ organization empty$
    'skip$
    { type$ "misc" =
        { organization }
        { * " ; " * organization " (Veranst.)" *}
      if$
    }
  if$
}

FUNCTION { format.tr.institution }
{ institution empty$
    'skip$
   { institution capitalize }
 if$
}

FUNCTION {format.ed.incoll}
{ editor empty$
     { "" }
     { " (Hrsg.)" 'fkt :=
       editor format.names
       format.editors.organization
     }
  if$
}

FUNCTION {article.in.journal}
{ duplicate$ empty$
    { pop$ "" }
    { author missing$ title missing$ and
            { emphasize }
            { emphasize "{In: }" swap$ * }
          if$
     }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { title }  %% Text so wie er dasteht im Feld title
  if$
}

FUNCTION {format.number}
{ number empty$
    { "" }
    { number " " * }  %% Text so wie er im Feld number steht plus " "
  if$
}


FUNCTION {format.digital.type}
{ type empty$
    { "" }
    { type #-1 #4 substring$ "mail" =
      type #1 #4 substring$ "Mail" =
      OR
        { "" }
        { type }  %% Typ einer digitalen Ressource in Form einer
              %% "Ergaenzenden Angabe", so wie er dasteht;
              %% Alternativ kann dieser Text auch in NOTE erfasst werden.
      if$
    }
 if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

%%  Auflagenvermerke gibt man komplett, einschliesslich Abkuerzungen in
%%  das Feld edition ein: ---> EDITION= { 3., erw. und verb. Aufl. }
%%  oder fremdsprachlich:      EDITION= { 2nd edition }

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { edition }
  if$
}

FUNCTION {format.date}
{ year duplicate$ empty$
    { "empty year in " cite$ * warning$
       pop$ "" }
     'skip$
  if$
  month empty$
     'skip$
    { type$ "book" =
      type$ "inbook" =
       OR
         'skip$
         { month " " * swap$ * }
      if$
     }
   if$
}


FUNCTION {format.edition.or.date}
{ edition empty$ year empty$ and
    { "" }
    { edition empty$
        { type empty$ NOT
            { type #-1 #4 substring$ "mail" =
              type #1 #4 substring$ "Mail" =
              OR
                { " gesendet: " "-- " type * swap$ *
                  format.date *
                }
                { "\,Version:\,"
                  format.date *
                }
              if$
             }
             { "\,Version:\,"
               format.date *
             }
           if$
        }
        { year empty$
            { "\,Version:\," edition * }
            { "\,Version:\," edition * ", " *
              format.date *
            }
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.doi}
{ doi empty$
    { "" }
    { new.block "\url{http://dx.doi.org/" doi * "}" *
     %%      { new.block "\url{http://dx.medra.org/" doi * "}" *
    }
  if$
}

FUNCTION {format.url}
{ urn missing$
     { doi missing$
          { url empty$
             { "" }
             { type empty$ NOT
                 { type #-1 #4 substring$ "mail" =
                   type #1 #4 substring$ "Mail" =
                   OR
                      { type$ "incollection" =
                          { "" }
                          { "\,Absenderadresse: \url{" url * "}" * }
                        if$
                      }
                      { "\url{" url * "}" * }%%   evtl. "URL" oder "<...>"
                    if$
                  }
                  { "\url{" url * "}" * }   %%   evtl. "URL" oder "<...>"
                if$
             }
           if$
          }
          { format.doi }
        if$
      }
      { "\url{http://nbn-resolving.de/urn/resolver.pl?urn=" urn * "}" *
      }
    if$
}

FUNCTION {format.maillist.url}
  { url empty$
       { "" }
       { type empty$
             { "" }
             { type #-1 #4 substring$ "mail" =
               type #1 #4 substring$ "Mail" =
               OR
                  { "\url{" url * "}" * }
                  { "" }
                if$
              }
           if$
        }
      if$
   }


FUNCTION {format.version.url}
{ url empty$ doi empty$ urn empty$ and and
   { type$ "techreport" =
       { format.edition }
       { "" }
     if$
   }
   { format.edition.or.date output format.url }
 if$
}

FUNCTION {format.edition.or.version}
{ url empty$ doi empty$ urn empty$ and and
   { format.edition }
   { format.edition.or.date }
 if$
}

FUNCTION {format.online.lastcheck}
{ lastchecked empty$
      { url empty$ doi empty$ urn empty$ and and
          { skip$ }
          { "" output }
        if$
      }
      { url empty$ doi empty$ urn empty$ and and
          { "there's a lastchecked date but no url, urn or doi in "
             cite$ * warning$
          }
          { part.of.sentence
            lastchecked "Abruf: " swap$ * output
          }
        if$
      }
    if$
}


FUNCTION {format.maillist.lastcheck}
   { type empty$ NOT
       { type #-1 #4 substring$ "mail" =
         type #1 #4 substring$ "Mail" =
         OR
           { format.online.lastcheck }
           'skip$
         if$
       }
       'skip$
     if$
   }

FUNCTION {format.address.publisher.year}
{  publisher empty$
      { address empty$
          { year empty$
              { "" }
              { year }
            if$
          }
          { "there's an address but no publisher in " cite$ * warning$
              address ", " * format.date *
          }
        if$
      }
      { address empty$
          { year empty$
              { "neither address nor publication date in " cite$ * warning$
                 publisher
              }
              { publisher ", " * format.date * }
            if$
          }
          { year empty$
              { address " : " * publisher * }
              { address " : " * publisher * ", " * format.date * }
            if$
          }
        if$
       }
     if$
}

FUNCTION {format.address.publisher.year.alt}
{  publisher empty$
      { address empty$
          { year empty$
              { "" }
              { url empty$
                   { year }
                   { "" }
                 if$
              }
            if$
          }
          { "there's an address but no publisher in " cite$ * warning$
              address ", " * format.date *
          }
        if$
      }
      {  url empty$%%%%  wenn es URL gibt wird nur die service-provider Adresse
%%%                      ausgegeben, die im publisher-Feld steht
          { address empty$
               { year empty$
                   { "neither address nor publication date in " cite$ * warning$
                      publisher
                   }
                   { publisher ", " * format.date * }
                 if$
               }
               { year empty$
                   { address " : " * publisher * }
                   { address " : " * publisher * ", " * format.date * }
                 if$
               }
             if$
          }
          { publisher }
        if$
      }
    if$
}

FUNCTION {format.howpublished}
{ url missing$ urn missing$ doi missing$ AND AND
    { howpublished empty$
        { address empty$
            { type empty$
              { "" }
              { type #-1 #4 substring$ "mail" =
                  { "(gesendet: " new.sentence
                    format.date * ")" *
                  }
                  { "" }
                if$
               }
             if$
             format.date *
            }
            { address ", " * format.date * }
          if$
        }
        { address empty$
            { howpublished ", " * format.date * }
            { address " : " * howpublished * ", " * format.date * }
          if$
        }
      if$
    }
    { howpublished empty$
        { "" }
        { howpublished }
      if$
    }
  if$
}

FUNCTION {format.lastchecked.or.type}%% nur in misc-Funktion
{  lastchecked empty$
      { url empty$ doi empty$ urn empty$ and and
          { type empty$
             { skip$ }
             { type set.period.dash.check
               type output
             }
           if$
          }
          { type empty$
              { skip$ }
              { type #-1 #4 substring$ "mail" =
                type #1 #4 substring$ "Mail" =
                OR
                  { skip$ }
                  { type set.period.dash.check
                         type output }
                     if$
                   }
                 if$
               }
             if$
           }
       { url empty$ doi empty$ urn empty$ and and
          { "there's a lastchecked date but no url, urn or doi in "
             cite$ * warning$
          }
          { type empty$
               { "there's a URL and a lastchecked date but no type in "
                  cite$ * warning$
               }
               { type set.period.dash.check
                 type output
                 part.of.sentence
                 lastchecked "Abruf: " swap$ * output
               }
             if$
          }
        if$
       }
    if$
}


FUNCTION {format.btitle}
{ title emphasize
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {format.btitle.vol}
{ number empty$
   { series empty$
     { volume empty$
        { title emphasize }
        { title emphasize ". Bd." * volume tie.or.space.connect }
       if$
     }
     { volume empty$
        { title emphasize }%% ein Buch, das zusaetzl. SERIES=Reihentitel besitzt
    %% jetzt kommt d. Fall des mehrbaendigen Werkes mit Gesamttitel=SERIES
    %% Zaehlung=VOLUME und Bandtitel=TITLE;
        { series emphasize ". Bd." * volume tie.or.space.connect
                                 ": " * "{\emph{" * title * "}}" * }
       if$
     }
     if$%% series-test
   }
   { title emphasize }%% wenn number ex., dann immer title-Feld
  if$%% Ende number-test
}

FUNCTION {format.series.number.din}
{ volume empty$
    { number empty$
          { series empty$
               { "" }
               { "(" series * ")" * } %% d. Seriennr koennte auch gleich hier
                                      %% im SERIES-Feld miterfasst werden
            if$
          }
          { series empty$
               { "(" number * ")" *
                 "there's a number but no series in " cite$ * warning$
               }
               { "(" series * number tie.or.space.connect ")" * }
            if$
          }
       if$
     }
     { series empty$
          { "" }
          { type$ "proceedings" =  %% Sonderfall, es darf VOLUME und NUMBER ex. !
            type$ "inproceedings" = OR
                { number empty$
                    { "(" series * ")" * }
                    { "(" series * number tie.or.space.connect ")" * }
                  if$
                }
                { "" }%%   Ausstieg mit Nullstring, s. Kommentar
            if$
          }%% bei gezaehlten Reihen MUSS die Reihennr. im Feld NUMBER stehen!
       if$ %% wenn also d. Feld VOLUME nicht leer ist, dann liegt ausser bei
           %% Typ PROCEEDINGS falsche
      }     %% Erfassung vor und es erfolgt d. Ausstieg mit d. Nullstring!
  if$
}

FUNCTION {format.tr.series.or.number}
{    number empty$
       { series empty$
           { "" }
           { "(" series * ")" * }
         if$
       }
       { series empty$
         { "(" number * ")" * }
         { "(" series * number tie.or.space.connect ")" * }
       if$
       }
     if$
 }

FUNCTION {format.misc.series}
{ series empty$
    { "" }
    { "(" series * ")" * }
  if$
}

FUNCTION { format.doi.urn }
{ urn empty$
     { doi empty$
          { "" }
          { "DOI" doi n.dashify tie.or.space.connect }
       if$
     }
     { "URN" urn n.dashify tie.or.space.connect }
  if$
}

FUNCTION { format.isbn.issn }
{ isbn empty$
     { issn empty$
          { "" }
          { "ISSN" issn n.dashify tie.or.space.connect }
       if$
     }
     { "ISBN" isbn n.dashify tie.or.space.connect }
  if$
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { url empty$
        { "S." pages n.dashify  tie.or.space.connect }
        { pages }
      if$
    }
  if$
}

FUNCTION {format.pages.book}
{ pages empty$
    { "" }
    { note empty$ isbn empty$ AND
         { "" pages n.dashify tie.or.space.connect " S" *
           add.period$ }
         { "" pages n.dashify tie.or.space.connect " S" * }
       if$
     }
  if$
}

FUNCTION {format.pages.bkcollation}
{ pages empty$
    { "" }
    { "" pages n.dashify tie.or.space.connect }
  if$
}

FUNCTION {format.bkpages.collat.check}
{ 's :=
  #1 'ptr :=
  s text.length$ 'collation :=
   collation #1 =
    { format.pages.book }
    {
      collation 'collrest :=
       { collrest #0 > }
       { s ptr #2 substring$ 't :=
         t "S." =
           { format.pages.bkcollation
             #0 'collrest := }
           { ptr #1 + 'ptr :=
             collrest #1 - 'collrest :=
             #1 collrest =
               { format.pages.book }
               { skip$ }
             if$
           }
         if$
       }
       while$
    }
  if$
}

FUNCTION {format.vol.year.num.pages}
{ volume field.or.null
    year empty$
      { "there's no year in " cite$ * warning$ }
      { " (" year * ")" * * }
    if$
    month empty$
      'skip$
      { ", " month * * }
    if$
    number empty$
      'skip$
      { ", Nr. " number * * }
    if$
    pages empty$
        'skip$
      { duplicate$ empty$
        { pop$ "" }
        { title missing$
            { ", " pages format.bkpages.collat.check * *}
            { ", " format.pages * *}
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.chapter.inbook}
{ duplicate$  empty$
    { pop$ "empty chapter in " cite$ * warning$ }
    { type empty$
         { "\emph{Kapitel\/} " swap$ tie.or.space.connect }
         { type " " * swap$ * }%%  wenn keine bes. Abschnittsform gen. werden soll,
                 %%  koennte e. kl. Zwischenraum gewaehlt werden, z.B. " \, "
       if$
     }
  if$
 }

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { "Kapitel " }
        { url empty$
           { type }
           { "Kapitel " }
         if$
        }
      if$
      chapter tie.or.space.connect
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.in.ed.booktitle.din}
{ booktitle empty$
    { "" }
    { editor empty$
        { volume empty$
            { "{In: }" booktitle emphasize * }
            { "{In: }" booktitle emphasize *
              " Bd." volume tie.or.space.connect *
            }
          if$
        }
        { volume empty$
            { "{In: }" format.ed.incoll * ": " * booktitle emphasize * }
            { "{In: }" format.ed.incoll * ": " * booktitle emphasize *
              " Bd." volume tie.or.space.connect *
            }
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.thesis.tr.type}
{ type empty$
    'skip$
   { pop$
     type
   }
 if$
}

FUNCTION {format.article.crossref}
{ key empty$
    { journal empty$
        { "need key or journal for " cite$ * " to crossref " * crossref *
          warning$
          ""
        }
        { "{In: }{\emph " journal * "}" * }
      if$
    }
    { "{In: }" key * }
  if$
  "\cite{" * crossref * "}" * ", " * format.pages *
}

FUNCTION {format.crossref.editor}
 { editor #1 "{ll}" format.name$ " (Hrsg.)" *
   editor num.names$ duplicate$
   #2 >
    { pop$ ua.etal * }%% --->u. a.
    { #2 <
        'skip$
        { editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            { ua.etal }
            { " ; " * editor #2 "{vv~}{ll}" format.name$ * " (Hrsg.)" * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.inbk.vol.title}
{ volume empty$
     { " In: " }
     { title empty$
        { " In: Bd." volume tie.or.space.connect
                " von " *
        }
        { "In: Bd." volume tie.or.space.connect ": " * title emphasize *
                     " (" * year * ") in " *
        }
      if$
     }
   if$
 }


FUNCTION {format.book.crossref}
{ type$ "inbook" =
   { format.inbk.vol.title }
   { volume empty$
       { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
         " "
       }
       { ". -- Bd." volume tie.or.space.connect
          " von " *
       }
     if$
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { series empty$
            { "need editor, key, or series for " cite$ * " to crossref " *
              crossref * warning$
              "" *
            }
            { "" * }
          if$
        }
        { key * }
      if$
    }
    { "" * }
  if$
  "\cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{ editor empty$
    editor field.or.null author field.or.null =
    or
     { key empty$
        { booktitle empty$
            { "need editor, key, or booktitle for " cite$ * " to crossref " *
              crossref * warning$
              ""
            }
            { "{In: }{\emph " booktitle * "}" * }%%
          if$
        }
        { "{In: }" }
      if$
     }
    { "{In: }" }
  if$
  "\cite{" * crossref * "}" * %% ", " * format.pages *    5.12.2005
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.title "title" output.check
  crossref missing$
    { journal article.in.journal output.nonnull
      new.sentence
      format.vol.year.num.pages output
      format.url output
    }
    { format.article.crossref output.nonnull }
  if$
  format.online.lastcheck
  doi set.period.dash.check
  urn set.period.dash.check
  format.doi.urn output
  issn set.period.dash.check
  format.isbn.issn output
  note set.period.dash.check
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors format.editors output.nonnull }
  if$
  set.colon.after
  crossref missing$
    { format.btitle.vol "title" output.check }
    { format.btitle "title" output.check }
  if$
  format.edition output
  format.address.publisher.year "publisher" output.check
  new.sentence
  crossref missing$
    { format.series.number.din output }
    { format.book.crossref output.nonnull }
  if$
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
  format.doi output
  format.url output
  new.block
  isbn set.period.dash.check
  format.isbn.issn output
  note set.period.dash.check
  note output
  fin.entry
}

FUNCTION {booklet}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors format.editors output.nonnull }
  if$
  set.colon.after
  format.btitle "title" output.check
  format.edition.or.version output
  format.url output
  format.online.lastcheck
  format.howpublished output
  series new.sentence.checka
  format.series.number.din output
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
  type set.period.dash.check
  format.digital.type output
  doi set.period.dash.check
  urn set.period.dash.check
  format.doi.urn output
  note set.period.dash.check
  note output
  format.isbn.issn output
  fin.entry
}


FUNCTION {inbook}
{ output.bibitem
    chapter format.chapter.inbook output.nonnull
      crossref missing$
      { author empty$
          { format.editors "\,{In:\,}" swap$ * "author and editor" output.check }
          { format.authors "\,{In:\,}" swap$ * output.nonnull }
        if$
        author empty$ editor empty$ AND
           { before.all 'output.state := }
           { set.colon.after }
         if$
      format.btitle.vol "title" output.check
      format.edition output
      format.address.publisher.year "publisher" output.check
      new.sentence
      format.series.number.din output
      isbn set.period.dash.check
      format.isbn.issn output
    }
    { format.book.crossref output.nonnull }
  if$
    part.of.sentence
    format.pages output
    note set.period.dash.check
    note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.title "title" output.check
  format.version.url output
  type empty$ NOT
      { type #-1 #4 substring$ "mail" =
        type #1 #4 substring$ "Mail" =
        OR
           'skip$
           { format.online.lastcheck }
        if$
      }
      { format.online.lastcheck }
   if$
  crossref missing$
    { format.in.ed.booktitle.din "booktitle" output.check
      format.edition output
      format.address.publisher.year "publisher" output.check
      format.maillist.url output
      format.maillist.lastcheck
      new.sentence
      format.series.number.din output
      doi set.period.dash.check
      urn set.period.dash.check
      format.doi.urn output
      isbn set.period.dash.check
      issn set.period.dash.check
      format.isbn.issn output
    }
    { format.incoll.inproc.crossref output.nonnull }
  if$
  part.of.sentence
  format.chapter.pages "pages" output.check
  note set.period.dash.check
  note output
  fin.entry
}


FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.title "title" output.check
  crossref missing$
    { format.in.ed.booktitle.din "booktitle" output.check
      address empty$
        { organization new.sentence.checka
          organization output
          part.of.sentence
          format.address.publisher.year output
        }
        { format.address.publisher.year "publisher" output.check }
      if$
      new.sentence
      series empty$
        'skip$
        { format.series.number.din output }
      if$
      isbn set.period.dash.check
      issn set.period.dash.check
      format.isbn.issn output
    }
    { format.incoll.inproc.crossref output.nonnull }
  if$
  part.of.sentence
  format.pages "pages" output.check
  note set.period.dash.check
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }%% nach Patashnik, wg US-Kompatibilitaet

FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization empty$
        { title empty$
            'skip$
            {format.btitle "title" output.check }
          if$
        }
        'skip$
      if$
    }
    'skip$
  if$
  format.authors.organization output.nonnull
  set.colon.after
  format.btitle "title" output.check
  format.edition "edition" output.check
  author empty$ organization empty$ AND
    {   address "address" output.check
        part.of.sentence
    }
    { organization empty$
        { address "address" output.check
          part.of.sentence
        }
        { address ": " * organization * output
          part.of.sentence
        }
      if$
    }
  if$
  format.date output
  number empty$
    'skip$
   { "(" number * ") " * output }
  if$
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
  format.doi output
  format.url output
  format.online.lastcheck
  note set.period.dash.check
  note output
  fin.entry
}

%%  Standard ist "Diplomarbeit", anderes mit TYPE="anderer Typ" erfassen!
%%  z.B. TYPE={Hausarbeit}, TYPE={Diss.}, TYPE={Habil.}, TYPE={Magisterarb.}
FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.btitle "title" output.check
  address output
  part.of.sentence
  school "school" output.check
  part.of.sentence
  "Diplomarbeit" format.thesis.tr.type output.nonnull
  part.of.sentence
  format.date "year" output.check
  format.url output
  format.online.lastcheck
%%  format.digital.resource.type
  doi set.period.dash.check
  urn set.period.dash.check
  format.doi.urn output
%%  pages set.period.dash.check
%%  format.pages.book output
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
  note set.period.dash.check
  note output
  fin.entry
}

FUNCTION {phdthesis} %%  {mastersthesis}% ist identisch bis auf Standardwert, s.o.
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.btitle "title" output.check
  address output
  part.of.sentence
  school "school" output.check
  part.of.sentence
  "Diss." format.thesis.tr.type output.nonnull % koennte auch `Dissertation' sein
  part.of.sentence
  format.date "year" output.check
  format.url output
  format.online.lastcheck
%%  format.digital.resource.type
  doi set.period.dash.check
  urn set.period.dash.check
  format.doi.urn output
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
%%  pages set.period.dash.check
%%  format.pages.book output
  note set.period.dash.check
  note output
  fin.entry
}


%% Normen, Vornormen, Schutzrechte (Patente) werden hiermit erfasst;
%% (z.Zt. auch noch Internetressourcen)
%% das type-Feld uebernimmt eine wichtige Steuerfunktion:
FUNCTION {misc}
{ output.bibitem
   type missing$ not
       { type duplicate$ #1 #4 substring$ "Norm" =
           type #1 #4 substring$ "Vorn" = OR
           { " " *
             format.number * output
             new.sentence
             format.date output
             title empty$
               { skip$ }
               { add.period$ new.sentence }
             if$
             format.btitle "title" output.check
             note set.period.dash.check
             note output
           }
           { duplicate$ #1 #6 substring$ "Schutz" =
                { " " * format.number * output
                  new.sentence
                  "(" * format.date ")" * output
                  add.period$ new.sentence
                  format.authors.organization add.period$ output
                  note output
                }
            %%  wenn irgendein anderer Typ eingetragen ist
                { pop$ pop$ ""
                  title empty$
                     { note empty$
                         { url empty$
                             { "there's no relevant field in " cite$ warning$
                                pop$ ""
                             }
                             { format.url output }%%%
                           if$
                         }
                         { note " " * output.nonnull }
                       if$
                     }
                     { author empty$
                          { editor empty$
                               { organization empty$
                                   { skip$ }
                                   { format.editors.organization output.nonnull
                                     set.colon.after
                                   }
                                 if$
                               }
                               { format.editors format.editors.organization
                                 output.nonnull set.colon.after
                               }
                             if$
                          }
                          { format.authors format.editors output.nonnull
                            set.colon.after
                          }
                        if$
                 format.btitle output.nonnull
                 url empty$
                   { format.edition output
                     format.howpublished output}
                   { format.howpublished output
                     format.edition.or.date output
                     format.url output
                   }
                 if$
                 format.online.lastcheck
%%                 format.lastchecked.or.type
                 new.sentence
                 format.misc.series output
%%                 note set.period.dash.check
                 note output
                }
              if$
            }
         if$
          }
        if$
       }
       %% wenn es keinen type gibt
       { title empty$
             { note empty$
                 { url empty$
                     { "there's no relevant field in " cite$ warning$
                       pop$ ""
                     }
                     { format.url output }%%%  das waere e. reine URL
                   if$
                 }
                 { note " " * output.nonnull
                 %%  format.url format.date output
                 }
               if$
             }
             { author empty$
                   { editor empty$
                        { organization empty$
                             { skip$ }
                             { format.editors.organization output.nonnull
                               set.colon.after
                             }
                           if$
                        }
                        { format.editors format.editors.organization
                          output.nonnull set.colon.after
                        }
                      if$
                   }
                   { format.authors format.editors output.nonnull
                     set.colon.after
                   }
                 if$
              format.btitle output.nonnull
              url empty$
                { format.edition output
                  format.howpublished output}
                { format.howpublished output
                  format.url output
                  format.edition.or.date output
                  format.online.lastcheck
%%                  format.lastchecked.or.type
                }
              if$
              new.sentence
              format.misc.series output
              note set.period.dash.check
              note output
          }
        if$
      }
    if$
 fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
    editor empty$
      { organization empty$
           { "empty organization and editor in " cite$ * warning$ }
           { organization " (Veranst.)" * output }
        if$
      }
      { format.editors format.editors.organization output.nonnull }
    if$
  set.colon.after
  format.btitle "title" output.check
  volume empty$
    { skip$ }
    { "{\textnormal{Bd.}}" volume tie.or.space.connect emphasize "volume" output.check }
  if$
  format.address.publisher.year "publisher" output.check
  new.sentence
  format.series.number.din output.nonnull
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
  isbn set.period.dash.check
  issn set.period.dash.check
  format.isbn.issn output
  note set.period.dash.check
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      format.tr.institution output.nonnull }
    { format.authors  format.editors output.nonnull }
  if$
  set.colon.after
  format.title "title" output.check
  institution empty$
     'skip$
    { author empty$ editor empty$ AND
          'skip$
        { institution new.sentence.checka
          "/ " institution *  output.nonnull
        }
      if$
    }
  if$
  format.version.url output
  format.online.lastcheck
  format.address.publisher.year output
  number new.sentence.checka
  format.tr.series.or.number "number" output.check
  "Forschungsbericht" format.thesis.tr.type set.period.dash.check
  "Forschungsbericht" format.thesis.tr.type output
%%  format.digital.resource.type
  pages empty$
    { skip$ }
    { pages set.period.dash.check
      pages format.bkpages.collat.check output
    }
  if$
  isbn set.period.dash.check
  issn set.period.dash.check
  format.isbn.issn output
  note set.period.dash.check
  note "note" output.check
  fin.entry
}

FUNCTION {unpublished} {misc}%%       AUTHOR, TITLE, NOTE muessen sein!
%%                                    andere Felder sind optional

FUNCTION {default.type} { misc }


MACRO {jan} {"Januar"}

MACRO {feb} {"Februar"}

MACRO {mar} {"M{\^^b a}rz"}

MACRO {apr} {"April"}

MACRO {mai} {"Mai"}

MACRO {may} {"Mai"}

MACRO {jun} {"Juni"}

MACRO {jul} {"Juli"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {okt} {"Oktober"}

MACRO {oct} {"Oktober"}

MACRO {nov} {"November"}

MACRO {dez} {"Dezember"}

MACRO {dec} {"Dezember"}

%%$$$  stillgelegte Beispiele fuer den Gebrauch von Kuerzeln (hier Zs-Titel).

%%MACRO {acmcs} {"ACM Computing Surveys"}

%%MACRO {acta} {"Acta Informatica"}

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}

FUNCTION {initialize.et.al.char.used}
{ #0 'et.al.char.used :=
}

EXECUTE {initialize.et.al.char.used}

FUNCTION {format.lab.names}
{ 's :=
  s num.names$ 'numnames :=
  numnames #1 >
    { numnames #4 >
        { #3 'namesleft := }
        { numnames 'namesleft := }
      if$
      #1 'nameptr :=
      ""
        { namesleft #0 > }
        { nameptr numnames =
            { s nameptr "{ff }{vv }{ll}{ jj}" format.name$ "others" =
%%                { "\," *  %% kein besonderes Zeichen fuer "others" i. label
                { "{\etalchar{+}}" *   %% ein plus-Zeichen (+) fuer "others"!
                  #1 'et.al.char.used :=
                }
                { s nameptr "{l{}}" format.name$ * }
              if$
            }
            { s nameptr "{l{}}" format.name$ * }
          if$
          nameptr #1 + 'nameptr :=
          namesleft #1 - 'namesleft :=
        }
      while$
      numnames #4 >
%%        { "\," *  %%  s. Bemerkung oben
        { "{\etalchar{+}}" *
          #1 'et.al.char.used :=
        }
        'skip$
      if$
    }
    { s #1 "{l{}}" format.name$
      duplicate$ text.length$ #2 <
        { pop$ s #1 "{ll}" format.name$ #3 text.prefix$ } %%  vgl. Anmerkung!
        'skip$
      if$
    }
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        { key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
        { key empty$
            { cite$ #1 #3 substring$ }
            { key #3 text.prefix$ }
          if$
        }
        { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.key.organization.label}
{ author empty$
    { key empty$
        { organization empty$
            { cite$ #1 #3 substring$ }
            { "The " #4 organization chop.word #3 text.prefix$ }
          if$
        }
        { key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.organization.label}
{ editor empty$
    { key empty$
        { organization empty$
            { cite$ #1 #3 substring$ }
            { "The " #4 organization chop.word organization }
          if$
        }
        { key #5 text.prefix$ }%%  man kann Laenge des key einstellen
      if$
    }
    { key empty$%%  wenn key vh., dann macht er das label! Lo,18/5/99
        { editor format.lab.names }
        { key #5 text.prefix$ }
      if$
    }
  if$
}

FUNCTION {calc.label}
{ type$ "book" =
  type$ "booklet" =
  type$ "inbook" =
  or or
    'author.editor.key.label
    { type$ "proceedings" =
        'editor.key.organization.label
        { type$ "manual" =
            'author.key.organization.label
            'author.key.label
          if$
        }
      if$
    }
  if$
  duplicate$
  year field.or.null purify$ #-1 #2 substring$
  *
  'label :=
  year field.or.null purify$ #-1 #4 substring$
  *
  sortify 'sort.label :=
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
        { "   " * }
        'skip$
      if$
      s nameptr "{ll{ }}{  ff{ }}{ vv{ }}{  jj{ }}" format.name$ 't :=
      nameptr numnames = t "others" = and
        { ua.etal * }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "Der " #4
        "Die " #4
          "Das " #4
            "Ein " #4
              "Eine " #5
                  "The " #4 t chop.word
               chop.word
             chop.word
           chop.word
         chop.word
       chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
        { key empty$
            { "to sort, need author, organization, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
        { key empty$
            { "to sort, need editor, organization, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}
{ calc.label
  sort.label
  "    "
  *
  type$ "book" =
  type$ "booklet" =
  type$ "inbook" =
  or or
    'author.editor.sort
    { type$ "proceedings" =
        'editor.organization.sort
        { type$ "manual" =
            'author.organization.sort
            'author.sort
          if$
        }
      if$
    }
  if$
  *
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #0 int.to.chr$ 'last.sort.label :=
  "" 'next.extra :=
  #0 'longest.label.width :=
  #0 'last.extra.num :=
}

FUNCTION {forward.pass}
{ last.sort.label sort.label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      sort.label 'last.sort.label :=
    }
  if$
}

FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  label extra.label * 'label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
  extra.label 'next.extra :=
}

EXECUTE {initialize.longest.label}

ITERATE {forward.pass}

REVERSE {reverse.pass}

FUNCTION {begin.bib}
{ et.al.char.used
    { "\newcommand{\etalchar}[1]{$^{#1}$}" write$ newline$ }
    'skip$
  if$
  preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
  newline$
  "% this bibliography is generated by alphadin.bst [8.2] from 2005-12-21"
  write$ newline$
  newline$
  "\providecommand{\url}[1]{\texttt{#1}}"
  write$ newline$
  "\expandafter\ifx\csname urlstyle\endcsname\relax"
  write$ newline$
  "  \providecommand{\doi}[1]{doi: #1}\else"
  write$ newline$
  "  \providecommand{\doi}{doi: \begingroup \urlstyle{rm}\Url}\fi"
  write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}